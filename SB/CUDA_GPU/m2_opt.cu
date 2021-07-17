#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "cublas_v2.h"
#include "cuda_profiler_api.h"

namespace cg = cooperative_groups;

// N defined here
#define N 2048
#define THREADS 512
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
    __syncthreads();
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
        lx = lx + deltaT * detune * ly;
        ly = ly - deltaT * (K * lx * lx + (detune - amplitude)) * lx;
    }

    x[idx] = lx;
    y[idx] = ly;
}

class Accelerator
{
private:
    float *couplings_pinned, *couplings_buf;
    uint *randvals_pinned, *randvals_buf;
    float *x_pinned, *x_buf, *y_pinned, *y_buf;
    int *spin_pinned, *spin_buf;
    int *E_buf, *E;

    cublasHandle_t handle;

    float coeff1 = DeltaT * xi;
    float coeff2 = 1.;
    int *results;
    dim3 grid, block;

    void init_coeff(std::string);
    void read_file(std::string);
    void init_randNum();
    void init_points();

    void step(int);

public:
    Accelerator(std::string);
    ~Accelerator();
    void run(int, int);
    void output(result *, std::string);
    void print_energyOnHost(float *, float *, float *, double, int);
};

void Accelerator::init_coeff(std::string fileName)
{
    gpuErrchk(cudaMallocHost((void **)&couplings_pinned, (N * N * sizeof(float))));
    memset(couplings_pinned, '\0', N * N * sizeof(float));
    gpuErrchk(cudaMalloc(&couplings_buf, N * N * sizeof(float)));

    gpuErrchk(cudaMallocHost((void **)&spin_pinned, (N * sizeof(int))));
    gpuErrchk(cudaMalloc(&spin_buf, (N * 16 * sizeof(int))));
    gpuErrchk(cudaMalloc(&E_buf, (N * sizeof(int))));
    gpuErrchk(cudaMallocHost((void **)&E, (N * sizeof(int))));

    read_file(fileName);

    gpuErrchk(cudaMemcpy(couplings_buf, couplings_pinned, N * N * sizeof(float), cudaMemcpyHostToDevice));
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
        couplings_pinned[a * N + b] = w;
        couplings_pinned[b * N + a] = w;
    }
    file.close();
}

void Accelerator::init_randNum()
{
    gpuErrchk(cudaMallocHost((void **)&randvals_pinned, (N * sizeof(uint))));
    gpuErrchk(cudaMalloc(&randvals_buf, N * sizeof(uint)));
    for (int i = 0; i < N; i++)
        randvals_pinned[i] = i;
    gpuErrchk(cudaMemcpy(randvals_buf, randvals_pinned, N * sizeof(uint), cudaMemcpyHostToDevice));
}

// initialize points
void Accelerator::init_points()
{
    gpuErrchk(cudaMallocHost((void **)&x_pinned, (N * 16 * sizeof(float))));
    gpuErrchk(cudaMalloc(&x_buf, N * 16 * sizeof(float)));
    gpuErrchk(cudaMallocHost((void **)&y_pinned, (N * 16 * sizeof(float))));
    gpuErrchk(cudaMalloc(&y_buf, N * 16 * sizeof(float)));
}

__global__ void calSpin(float *x_buf, int *spin_buf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    spin_buf[idx] = x_buf[idx] > 0.0 ? 1 : -1;
}

__device__ void energyReduce(int *E_buf, int idx, int i){
    if ((idx >> i) % 2 == 0) E_buf[idx] += E_buf[idx + (1 << i)];
}

__global__ void calEnergy(int *spin_buf, int *E_buf, float *couplings_buf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int E = 0;
    for (int i = 0; i < N; i++)
    {
        E += spin_buf[idx] * spin_buf[i] * couplings_buf[idx * N + i];
    }
    E_buf[idx] = E;
    __syncthreads();
    for (int i = 0; i < 15; i++)
    {
        energyReduce(E_buf, idx, i);
    }
}


// n step iter for a run
void Accelerator::run(int times, int steps)
{
    results = (int *)malloc(times * sizeof(int));
    for (int t = 0; t < times; t++)
    {
        prepare_points<<<grid, block>>>(x_buf, y_buf, randvals_buf);
        for (int s = 0; s < steps; s++)
            step(s);

        // // Get Result from device
        // gpuErrchk(cudaMemcpy(x_pinned, x_buf, N * sizeof(float), cudaMemcpyDeviceToHost));

        calSpin<<<grid, block>>>(x_buf, spin_buf);
        calEnergy<<<grid, block>>>(spin_buf, E_buf, couplings_buf);
        
        gpuErrchk(cudaMemcpy(E, E_buf, sizeof(int), cudaMemcpyDeviceToHost));

        int res=E[0];
        printf("%d\n", res);
        results[t] = res;
    }
}

void Accelerator::step(int i)
{
    float amplitude = i / (float)STEP;
    UpdateTwice<<<grid, block>>>(x_buf, y_buf, amplitude);
    cublasSgemv(handle, CUBLAS_OP_N, N, N, &coeff1,
                couplings_buf, N,
                x_buf, 1,
                &coeff2, y_buf, 1);
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

// void Accelerator::print_energyOnHost(float *x, float *x_buf, float *couplings, double curr, int t)
// {
//     gpuErrchk(cudaMemcpy(x, x_buf, N * sizeof(float), cudaMemcpyDeviceToHost));
//     int E = calEnergy();
//     printf("%d %lf %d\n", t, curr, -E);
// }

Accelerator::Accelerator(string fileName) : grid(N / THREADS), block(THREADS) //38 SMs on 3060Ti
{
    init_coeff(fileName);
    init_randNum();
    init_points();
    //cublas
    cublasCreate(&handle);
}

Accelerator::~Accelerator()
{
    cudaFreeHost(couplings_pinned);
    cudaFreeHost(x_pinned);
    cudaFreeHost(y_pinned);
    cudaFreeHost(randvals_pinned);
    cudaFreeHost(spin_pinned);
    cudaFreeHost(E);
    cudaFree(couplings_buf);
    cudaFree(x_buf);
    cudaFree(y_buf);
    cudaFree(randvals_buf);
    cudaFree(spin_buf);
    cudaFree(E_buf);
    free(results);
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

    
    double timer = 0.;
    float gpu_timer = 0;
    cudaEvent_t start, stop;

    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    gpuErrchk(cudaEventRecord(start, 0));
    clock_t begin = clock();

    acc.run(TIMES, STEP);

    clock_t end = clock();

    gpuErrchk(cudaEventRecord(stop, 0));
    gpuErrchk(cudaEventSynchronize(stop));

    gpuErrchk(cudaEventElapsedTime(&gpu_timer, start, stop));

    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));

    timer = (double)(end - begin) / CLOCKS_PER_SEC;
    gpu_timer /= 1000;
    printf("cpu elapse time:%6.6lf, gpu elapse time:%6.6lf\n", timer, gpu_timer);

    acc.output(results);
    return 0;
}
