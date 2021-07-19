#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "cublas_v2.h"
#include "cuda_profiler_api.h"
#include "gpuErrchk.cuh"

namespace cg = cooperative_groups;

// N defined here
#define N 32768
#define THREADS 512
#define TIMES 3
#define MAX 4294967295.0

// parameter settings (p > detune - xi, bifurcation point)
#define K 1.
#define detune (float)(1.)
#define xi 0.01565 // 0.7*detune / (rho * sqrt(N))
#define deltaT (float)(0.5)
#define DeltaT (float)(1.0)
#define M 2
#define STEP 100

using result = int;
using namespace std;

__device__ uint xorshift32(uint *state)
{
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

__global__ void prepare_points(float *x, float *y,
                               uint *randvals_buf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    x[idx] = ((xorshift32(&randvals_buf[idx]) / (float)MAX) / 10.0);
    y[idx] = ((xorshift32(&randvals_buf[idx]) / (float)MAX) / 10.0);
}


class Accelerator
{
public:
    float *couplings_buf;
    uint *randvals_buf;
    float *x_pinned, *x_buf, *y_buf, *z_buf;
    float *E_pinned, *spin_buf, *E_buf;

    cublasHandle_t handle;

    float coeff1 = DeltaT * xi;
    float coeff2 = 1.;
    float *results;
    dim3 grid, block;

    void init_coeff(std::string);
    void read_file(std::string, float *);
    void init_randNum();
    void init_points();

    void UpdateX();
    void UpdateY(float);
    void UpdateTwice(float);
    void step(int, int);

    void calSpinEnergy();

public:
    Accelerator(std::string);
    ~Accelerator();
    void run(int, int);
    void output(std::string);
    void print_energyOnHost(float *, float *, float *, double, int);
};

void Accelerator::init_coeff(std::string fileName)
{
    float *couplings_pinned;
    gpuErrchk(cudaMallocHost((void **)&couplings_pinned, (N * N * sizeof(float))));
    gpuErrchk(cudaMalloc(&couplings_buf, N * N * sizeof(float)));
    read_file(fileName, couplings_pinned);
    gpuErrchk(cudaMemcpy(couplings_buf, couplings_pinned, N * N * sizeof(float), cudaMemcpyHostToDevice));
    std::cout << std::endl;
    gpuErrchk(cudaFreeHost(couplings_pinned));

    gpuErrchk(cudaMallocHost((void **)&E_pinned, 16 * 16 * sizeof(float)));
    gpuErrchk(cudaMalloc(&spin_buf, (N * 16 * sizeof(float))));
    gpuErrchk(cudaMalloc(&E_buf, (16 * 16 * sizeof(float))));
}

// Read couplings file
void Accelerator::read_file(std::string fileName, float *couplings)
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
        couplings[a * N + b] = (float)w;
        couplings[b * N + a] = (float)w;
    }
    file.close();
}

void Accelerator::init_randNum()
{
    float *randvals_pinned;
    gpuErrchk(cudaMallocHost((void **)&randvals_pinned, (N * 16 * sizeof(uint))));
    for (int i = 0; i < 16 * N; i++)
        randvals_pinned[i] = i;
    gpuErrchk(cudaMalloc(&randvals_buf, N * 16 * sizeof(uint)));
    gpuErrchk(cudaMemcpy(randvals_buf, randvals_pinned, N * 16 * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrchk(cudaFreeHost(randvals_pinned));
}

// initialize points
void Accelerator::init_points()
{
    gpuErrchk(cudaMallocHost((void **)&x_pinned, (N * 16 * sizeof(float))));
    gpuErrchk(cudaMalloc(&x_buf, N * 16 * sizeof(float)));
    gpuErrchk(cudaMalloc(&y_buf, N * 16 * sizeof(float)));
    gpuErrchk(cudaMalloc(&z_buf, N * 16 * sizeof(float)));
}

__global__ void calSpin(float *x_buf, float *spin_buf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    spin_buf[idx] = x_buf[idx] > 0.0 ? (float)1 : (float)-1;
}

void Accelerator::calSpinEnergy()
{
    float alpha = 1.0, beta = 0.0;
    gpuErrchk(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          N, 16, N,
                          &alpha, couplings_buf, N,
                          spin_buf, N,
                          &beta, z_buf, N));
    // std::cout << "spin1 ok\n";
    gpuErrchk(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                          16, 16, N,
                          &alpha, spin_buf, N, // Warning: might not be N after transposed.
                          z_buf, N,
                          &beta, E_buf, 16));
    // std::cout << "spin2 ok\n";
}

__global__ void getEnergy(float *E_buf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < 16; i++)
    {
        if (idx == (i * 16 + i))
        {
            E_buf[i] = E_buf[idx];
            break;
        }
    }
}

// n step iter for a run
void Accelerator::run(int times, int steps)
{
    results = (float *)malloc(16 * times * sizeof(float));
    memset(results, 0, 16 * times * sizeof(float));

    for (int t = 0; t < times; t++)
    {
        std::cout << t << " times start\n";
        prepare_points<<<grid, block>>>(x_buf, y_buf, randvals_buf);

        for (int s = 0; s < steps; s++)
            step(s, steps);

        // // Get Result from device
        calSpin<<<grid, block>>>(x_buf, spin_buf);

        // cudaDeviceSynchronize();
        calSpinEnergy();
        gpuErrchk(cudaMemcpy(E_pinned, E_buf, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost));
        // cudaDeviceSynchronize();

        for(int i = 0; i < 16; i ++){
            for(int j = 0; j < 16; j++){
                std::cout << E_pinned[j*16 + i] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << "\n\n";

        for (int i = 0; i < 16; i++)
        {
            results[t * 16 + i] = E_pinned[i];
            std::cout << E_pinned[i] << ' ';
        }
        std::cout << std::endl;
        std::cout << t << " times end\n";
    }
}

__global__ void XYtoZ(float *x, float *y, float *z)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    z[idx] = x[idx] * y[idx];
}

void Accelerator::UpdateX()
{
    float alpha = deltaT * detune;
    cublasSaxpy(handle, 16 * N, &alpha, y_buf, 1, x_buf, 1);
}
void Accelerator::UpdateY(float amplitude)
{
    XYtoZ<<<grid, block>>>(x_buf, x_buf, z_buf);
    XYtoZ<<<grid, block>>>(z_buf, x_buf, z_buf);
    float alpha = (amplitude - detune) / (float)K;
    cublasSaxpy(handle, 16 * N, &alpha, x_buf, 1, z_buf, 1);
    alpha = -deltaT * K;
    cublasSaxpy(handle, 16 * N, &alpha, z_buf, 1, y_buf, 1);
}
void Accelerator::UpdateTwice(float amplitude)
{
    UpdateX();
    UpdateY(amplitude);
    UpdateX();
    UpdateY(amplitude);
}

void Accelerator::step(int i, int steps)
{
    float amplitude = (float)i / (float)steps;
    UpdateTwice(amplitude);
    cublasSgemv(handle, CUBLAS_OP_N,
                N, N, &coeff1, couplings_buf, N,
                x_buf, 1,
                &coeff2, y_buf, 1);
}

// Write statistics to file
void Accelerator::output(std::string outName = "output.txt")
{
    std::ofstream file;
    file.open(outName);
    for (int i = 0; i < 16 * TIMES; i++)
        file << results[i] << std::endl;
    file.close();
}

// void Accelerator::print_energyOnHost(float *x, float *x_buf, float *couplings, double curr, int t)
// {
//     gpuErrchk(cudaMemcpy(x, x_buf, N * sizeof(float), cudaMemcpyDeviceToHost));
//     int E = calEnergy();
//     printf("%d %lf %d\n", t, curr, -E);
// }

Accelerator::Accelerator(string fileName) : grid(16 * N / THREADS), block(THREADS)
{
    init_coeff(fileName);
    init_randNum();
    init_points();
    //cublas
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
}

Accelerator::~Accelerator()
{
    cublasDestroy(handle);
    cudaFreeHost(x_pinned);
    cudaFreeHost(E_pinned);
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

    acc.output();
    return 0;
}
