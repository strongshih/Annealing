#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda_profiler_api.h"

// N defined here
#define N 2048
#define THREADS 64
#define TIMES 10
#define MAX 4294967295.0

// parameter settings (p > detune - xi, bifurcation point)
#define K 1.
#define detune 1.
#define xi 0.01565 // 0.7*detune / (rho * sqrt(N))
#define deltaT 0.5
#define DeltaT 1.0
#define M 2
#define STEP 100

using result = int;
using namespace std;

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__device__ uint xorshift32(uint *state)
{
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

__global__ void prepare_points(float *x,
                               float *y,
                               uint *randvals_buf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    x[idx] = ((xorshift32(&randvals_buf[idx]) / (float)MAX) / 10.0);
    y[idx] = ((xorshift32(&randvals_buf[idx]) / (float)MAX) / 10.0);
}

__global__ void UpdateTwice(float *x,
                            float *y,
                            float amplitude)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float lx = x[idx];
    float ly = y[idx];

#pragma unroll
    for (int i = 0; i < M; i++)
    {
        lx = lx + detune * ly * deltaT;
        ly = ly - deltaT * (K * lx * lx * lx + (detune - amplitude) * lx);
    }

    x[idx] = lx;
    y[idx] = ly;
}

class Accelerator
{
private:
    float *couplings, *couplings_buf;
    uint *randvals_buf, *randvals;
    float *x, *x_buf, *y, *y_buf;

    cublasHandle_t handle;

    float coeff1 = DeltaT * xi;
    float coeff2 = 1.;
    dim3 grid, block;

    void init_coeff(std::string);
    void read_file(std::string);
    void init_randNum();
    void init_points();

    result calEnergy();
    void iter(int, double &);

public:
    Accelerator(std::string);
    ~Accelerator();
    int run(int);
    void output(result *, std::string);
    void print_energyOnHost(float *, float *, float *, double, int);
};

void Accelerator::init_coeff(std::string fileName)
{
    couplings = (float *)malloc(N * N * sizeof(float));
    memset(couplings, '\0', N * N * sizeof(float));
    gpuErrchk(cudaMalloc(&couplings_buf, N * N * sizeof(float)));

    read_file(fileName);

    gpuErrchk(cudaMemcpy(couplings_buf, couplings, N * N * sizeof(float), cudaMemcpyHostToDevice));
}

// Read couplings file
void Accelerator::read_file(std::string fileName)
{
    std::ifstream file;
    try
    {
        file.open(fileName);
    }
    catch (std::ios_base::failure &e)
    {
        std::cerr << e.what() << '\n';
        exit(0);
    }
    int a, b, w;
    file >> a;
    while (!file.eof())
    {
        file >> a >> b >> w;
        assert(a != b); // not dealing with external field
        couplings[a * N + b] = w;
        couplings[b * N + a] = w;
    }
    file.close();
}

void Accelerator::init_randNum()
{
    randvals = (uint *)malloc(N * sizeof(uint));
    gpuErrchk(cudaMalloc(&randvals_buf, N * sizeof(uint)));
    for (int i = 0; i < N; i++)
        randvals[i] = i;
    gpuErrchk(cudaMemcpy(randvals_buf, randvals, N * sizeof(uint), cudaMemcpyHostToDevice));
}

// initialize points
void Accelerator::init_points()
{
    x = (float *)malloc(N * sizeof(float));
    gpuErrchk(cudaMalloc(&x_buf, N * sizeof(float)));
    y = (float *)malloc(N * sizeof(float));
    gpuErrchk(cudaMalloc(&y_buf, N * sizeof(float)));
}

result Accelerator::calEnergy()
{
    int E = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = i + 1; j < N; j++)
        {
            int a = x[i] > 0.0 ? 1 : -1;
            int b = x[j] > 0.0 ? 1 : -1;
            E += a * b * couplings[i * N + j];
        }
    }
    return result(-E);
}

// n step iter for a run
int Accelerator::run(int n)
{
    prepare_points<<<grid, block>>>(x_buf, y_buf, randvals_buf);

    double timer = 0.;
    for (int i = 0; i < n; i++)
        iter(i, timer);

    printf("%lf\n", timer);

    // Get Result from device
    gpuErrchk(cudaMemcpy(x, x_buf, N * sizeof(float), cudaMemcpyDeviceToHost));

    return calEnergy();
}

void Accelerator::iter(int i, double &timer)
{
    float amplitude = i / (float)STEP;
    clock_t begin = clock();
    UpdateTwice<<<grid, block>>>(x_buf, y_buf, amplitude);
    cublasSgemv(handle, CUBLAS_OP_N, N, N, &coeff1, couplings_buf, N, x_buf, 1, &coeff2, y_buf, 1);
    clock_t end = clock();
    double duration = (double)(end - begin) / CLOCKS_PER_SEC;
    timer += duration;
}

// Write statistics to file
void Accelerator::output(result results[], std::string outName = "output.txt")
{
    std::ofstream file;
    file.open(outName);
    for (int i = 0; i < TIMES; i++)
        file << results[i] << std::endl;
    file.close();
}

void Accelerator::print_energyOnHost(float *x, float *x_buf, float *couplings, double curr, int t)
{
    gpuErrchk(cudaMemcpy(x, x_buf, N * sizeof(float), cudaMemcpyDeviceToHost));
    int E = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = i + 1; j < N; j++)
        {
            int a = x[i] > 0.0 ? 1 : -1;
            int b = x[j] > 0.0 ? 1 : -1;
            E += a * b * couplings[i * N + j];
        }
    }
    printf("%d %lf %d\n", t, curr, -E);
}

Accelerator::Accelerator(string fileName) : grid(N / THREADS), block(THREADS)
{
    init_coeff(fileName);
    init_randNum();
    init_points();
    //cublas
    cublasCreate(&handle);
}

Accelerator::~Accelerator()
{
    free(couplings);
    free(x);
    free(y);
    free(randvals);
    cudaFree(couplings_buf);
    cudaFree(x_buf);
    cudaFree(y_buf);
    cudaFree(randvals_buf);
}

void usage()
{
    printf("Usage:\n");
    printf("       ./Bifurcation-cuda [spin configuration]\n");
    exit(0);
}

int main(int argc, char *argv[])
{
    if (argc != 2)
        usage();

    std::string filename = argv[1];

    Accelerator acc(filename);

    result results[TIMES];

    for (int t = 0; t < TIMES; t++)
    {
        results[t] = acc.run(STEP);
    }

    acc.output(results);
    return 0;
}
