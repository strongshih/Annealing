#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAXDEVICE 10
#define MAXK 2048
#define N 4096
#define THREAD_NUM 1024
#define TIMES 1
#define NANO2SECOND 1000000000.0
#define ix threadIdx.x
#define iy threadIdx.x+1024

#define SWEEP 200
#define MAX 4294967295

#define FLOAT_MAX 4294967295.0f

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ unsigned int xorshift32(unsigned int* state)
{
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

__device__ void shadow_assignback(int* state, int* shadow_state, int n)
{
    state[n] = shadow_state[n];
}

__global__ void randnum_init(unsigned int* randnum)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    randnum[idx] = idx;
}

__global__ void spins_init(int* spins, int* shadow_spins)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int randnum = idx;
    spins[idx] = ((xorshift32(&randnum) & 1) << 1) - 1;
    shadow_spins[idx] = spins[idx];
}

__global__ void field_init(int* L, int* couplings, int* spins)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    L[idx] = 0;
    for (int j = 0; j < N; j++) {
        if (j == idx) {
            L[idx] += couplings[idx * N + j];
        }
        else {
            L[idx] += couplings[idx * N + j] * spins[j];
        }
    }
}

__device__ void field_update(int* F, bool* delta, int n, int* couplings, int* shadow_spins)
{
    for (int j = 0; j < N; j++) {
        if (j != n && delta[j]) {
            F[n] += 2 * couplings[n * N + j] * shadow_spins[j];
        }
    }
}


__global__ void ising(int* spins,
    int* shadow_spins,
    float beta,
    float q,
    int* L, int* couplings, bool* delta, int s, unsigned int* randnum)
{
    // random number
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //difference -> L[N]
    //Field initialize

    // annealing

    float r = 4 * (((float)xorshift32(&randnum[idx]) / FLOAT_MAX) - 0.5) / beta;
    delta[idx] = false;
    if ((L[idx] * spins[idx] + q / beta) > r) {
        shadow_spins[idx] = -spins[idx];
        delta[idx] = true;
    }
}
__global__ void update_phase(int* L, bool* delta, int* couplings, int* spins, int* shadow_spins) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // update field by delta
    field_update(L, delta, idx, couplings, shadow_spins);

    // update spins
    shadow_assignback(spins, shadow_spins, idx);
}

void usage() {
    printf("Usage:\n");
    printf("       ./Ising-opencl [spin configuration]\n");
    exit(0);
}

int main(int argc, char* argv[]) {
    //if (argc != 2)
    //    usage();

    // initialize parameters
    int* L, * couplings, * results;
    int* L_buf, * couplings_buf;
    bool* delta, * delta_buf;

    float beta = 0.1f; // from 0.1 to 3.0
    float beta_increase = (3.0f - 0.1f) / (float)SWEEP;
    float q = 0.0f; // from 0 to SWEEP
    float q_increase = 0.09f;
    int* spins, * shadow_spins;
    int* spins_buf, * shadow_spins_buf;
    unsigned int* randnum_buf;

    couplings = (int*)malloc(N * N * sizeof(int));
    results = (int*)malloc(TIMES * sizeof(int));
    delta = (bool*)malloc(N * sizeof(bool));
    spins = (int*)malloc(N * sizeof(int));
    shadow_spins = (int*)malloc(N * sizeof(int));
    L = (int*)malloc(N * sizeof(int));

    memset(couplings, '\0', N * N * sizeof(int));
    memset(results, '\0', TIMES * sizeof(int));
    memset(L, '\0', N * sizeof(int));
    memset(delta, '\0', N * sizeof(bool));

    gpuErrchk(cudaMalloc(&couplings_buf, N * N * sizeof(int)));
    //gpuErrchk(cudaMalloc(&results_buf, TIMES * sizeof(int)));
    gpuErrchk(cudaMalloc(&L_buf, N * sizeof(int)));
    gpuErrchk(cudaMalloc(&spins_buf, N * sizeof(int)));
    gpuErrchk(cudaMalloc(&shadow_spins_buf, N * sizeof(int)));
    gpuErrchk(cudaMalloc(&delta_buf, N * sizeof(int)));
    gpuErrchk(cudaMalloc(&randnum_buf, N * sizeof(unsigned int)));

    // Read couplings file 
    FILE* instance = fopen(argv[1], "r");
    //FILE* instance = fopen("input.txt", "r");
    
    assert(instance != NULL);
    int a, b, w;
    fscanf(instance, "%d", &a);
    while (!feof(instance)) {
        fscanf(instance, "%d%d%d", &a, &b, &w);
        couplings[a * N + b] = w;
        couplings[b * N + a] = w;
    }
    fclose(instance);
    
    printf("coupling [1][2000]: %d.\n", couplings[N+2000]);
    fflush(stdout);
    printf("file input done.\n");
    fflush(stdout);
    
    // copy couplings coefficients
    gpuErrchk(cudaMemcpy(couplings_buf, couplings, N * N * sizeof(int), cudaMemcpyHostToDevice));
    printf("couplings into GPU.\n");
    fflush(stdout);

    gpuErrchk(cudaMemcpy(delta_buf, delta, N * sizeof(bool), cudaMemcpyHostToDevice));
    printf("Local fields into GPU.\n");
    fflush(stdout);


    // launching kernel
    dim3 grid(N/THREAD_NUM), block(THREAD_NUM);
    
    printf("kernel launched.\n");
    fflush(stdout);

    randnum_init << <grid, block >> > (randnum_buf);
    for (int i = 0; i < TIMES; i++) {
        
        spins_init << <grid, block >> > (spins_buf, shadow_spins_buf);
        field_init << <grid, block >> > (L_buf, couplings_buf, spins_buf);
        clock_t begin = clock();
        for (int j = 0; j < SWEEP; j++) {
            //clock_t begin = clock();
            beta += beta_increase;
            q += q_increase;
            ising << < grid, block >> > (spins_buf, shadow_spins_buf, beta, q, L_buf, couplings_buf, delta_buf, j, randnum_buf);
            update_phase << < grid, block >> > (L_buf, delta_buf, couplings_buf, spins_buf, shadow_spins_buf);
            //clock_t end = clock();
            //printf("Sweep %d, time : %lld, %lld\n", j, begin, end);
        }
        clock_t end = clock();
        printf("Time : %lf\n", (double)(end-begin)/CLOCKS_PER_SEC);
        gpuErrchk(cudaMemcpy(spins, spins_buf, N * sizeof(int), cudaMemcpyDeviceToHost));
        field_init << <grid, block >> > (L_buf, couplings_buf, spins_buf);
        gpuErrchk(cudaMemcpy(L, L_buf, N * sizeof(int), cudaMemcpyDeviceToHost));
        int E = 0;
        for (int j = 0; j < N; j++) {
            E += L[j] * spins[j];
        }
        results[i] = -E/2;
    }
    
    // Get Result from device
    //gpuErrchk(cudaMemcpy(results, results_buf, TIMES * sizeof(int), cudaMemcpyDeviceToHost));

    // Write statistics to file
    FILE* output;
    output = fopen("output.txt", "w");
    int min_result = INT_MAX;
    for (int i = 0; i < TIMES; i++) {
        fprintf(output, "%d\n", results[i]);
        if (results[i] < min_result) min_result = results[i];
    }
    fclose(output);
    printf("file output done.\n");
    fflush(stdout);

    printf("min. result is %d.\n", min_result);
    fflush(stdout);
    // Release Objects
    free(results);
    free(couplings);
    free(L);
    free(spins);
    free(shadow_spins);
    cudaFree(couplings_buf);
    return 0;
}

//cudaError_t SCA();
