#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda_profiler_api.h"

#define N 32768
#define THREADS 64
#define TIMES 1
#define MAX 4294967295.0

// parameter settings (p > detune - xi, bifurcation point)
#define K 1.
#define detune 1.
#define xi 0.01565 // 0.7*detune / (rho * sqrt(N))
#define deltaT 0.5
#define DeltaT 1.0
#define M 2
#define STEP 100

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
                               uint *randvals)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    x[idx] = ((xorshift32(&randvals[idx]) / (float)MAX) / 10.0);
    y[idx] = ((xorshift32(&randvals[idx]) / (float)MAX) / 10.0);
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

void print_energy(float *x, float *x_buf, float *couplings, double curr, int t)
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

    // initialize couplings
    float *couplings, *couplings_buf;
    couplings = (float *)malloc(N * N * sizeof(float));
    memset(couplings, '\0', N * N * sizeof(float));
    gpuErrchk(cudaMalloc(&couplings_buf, N * N * sizeof(float)));

    // Read couplings file
    FILE *instance = fopen(argv[1], "r");
    assert(instance != NULL);
    int a, b, w;
    fscanf(instance, "%d", &a);
    while (!feof(instance))
    {
        fscanf(instance, "%d%d%d", &a, &b, &w);
        assert(a != b); // not dealing with external field
        couplings[a * N + b] = w;
        couplings[b * N + a] = w;
    }
    fclose(instance);

    // cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    // copy couplings to target device
    gpuErrchk(cudaMemcpy(couplings_buf, couplings, N * N * sizeof(float), cudaMemcpyHostToDevice));

    // initialize random number
    uint *randvals, *initRand;
    gpuErrchk(cudaMalloc(&randvals, N * sizeof(uint)));
    initRand = (uint *)malloc(N * sizeof(uint));
    for (int i = 0; i < N; i++)
        initRand[i] = i;
    gpuErrchk(cudaMemcpy(randvals, initRand, N * sizeof(uint), cudaMemcpyHostToDevice));

    // initialize points
    float *x, *x_buf, *y, *y_buf;
    x = (float *)malloc(N * sizeof(float));
    gpuErrchk(cudaMalloc(&x_buf, N * sizeof(float)));
    y = (float *)malloc(N * sizeof(float));
    gpuErrchk(cudaMalloc(&y_buf, N * sizeof(float)));

    // launching kernel
    float coeff1 = DeltaT * xi;
    float coeff2 = 1.;
    dim3 grid(N / THREADS), block(THREADS);
    int results[TIMES] = {0};
    for (int t = 0; t < TIMES; t++)
    {
        prepare_points<<<grid, block>>>(x_buf, y_buf, randvals);
        double curr = 0.;
        for (int s = 0; s < STEP; s++)
        {
            float amplitude = s / (float)STEP;
            clock_t begin = clock();
            float elapsed = 0;
            cudaEvent_t start, stop;

            gpuErrchk(cudaEventCreate(&start));
            gpuErrchk(cudaEventCreate(&stop));

            gpuErrchk(cudaEventRecord(start, 0));
            
            UpdateTwice<<<grid, block>>>(x_buf, y_buf, amplitude);
			cublasSgemv(handle, CUBLAS_OP_N, N, N, &coeff1, couplings_buf, N, x_buf, 1, &coeff2, y_buf, 1);

            gpuErrchk(cudaEventRecord(stop, 0));
            gpuErrchk(cudaEventSynchronize(stop));

            gpuErrchk(cudaEventElapsedTime(&elapsed, start, stop));

            gpuErrchk(cudaEventDestroy(start));
            gpuErrchk(cudaEventDestroy(stop));

            printf("The elapsed time in gpu was %.2f ms\n", elapsed);

			
			clock_t end = clock();
			double duration = (double)(end-begin) / CLOCKS_PER_SEC;
			curr += duration;
			printf("%lf\n", curr);
            // print_energy(x, x_buf, couplings, curr, t);
        }
        // Get Result from device
        gpuErrchk(cudaMemcpy(x, x_buf, N * sizeof(float), cudaMemcpyDeviceToHost));

        // calculate energy
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
        results[t] = -E;
    }

    // Write statistics to file
    FILE *output;
    output = fopen("output.txt", "w");
    for (int i = 0; i < TIMES; i++)
        fprintf(output, "%d\n", results[i]);
    fclose(output);

    // Release Objects
    free(couplings);
    free(x);
    free(y);
    free(initRand);
    cudaFree(couplings_buf);
    cudaFree(x_buf);
    cudaFree(y_buf);
    cudaFree(randvals);
    return 0;
}
