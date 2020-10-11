#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_awbarrier.h>
#include <cooperative_groups.h>


#define MAXDEVICE 10

#define GRID_SIZE N/CUDA_COUNT
#define CUDA_COUNT 2
#define MAXK 2048
#define N 2048

#define THREAD_NUM 1024
#define TIMES 10
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

__global__ void randnum_init(unsigned int* randnum, int offset)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    randnum[idx] = idx + offset;
}

__device__ void _spins_init(int** spins, int** shadow_spins, int n, int device_num)
{
    for(int i = 1; i < CUDA_COUNT; i++){
        spins[(device_num+i)%CUDA_COUNT][n] = spins[device_num][n];
        shadow_spins[(device_num+i)%CUDA_COUNT][n] = spins[device_num][n];
    }
}

__global__ void spins_init(int** spins, int** shadow_spins, unsigned int* randnum, int device_num, int offset)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    spins[device_num][offset + idx] = ((xorshift32(&randnum[idx]) & 1) << 1) - 1;
    shadow_spins[device_num][offset + idx] = spins[device_num][offset + idx];
    _spins_init(spins, shadow_spins, offset + idx, device_num);
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
    int cuda_count;
    gpuErrchk(cudaGetDeviceCount(&cuda_count));
    printf("number of devices: %2d", cuda_count);
    if(cuda_count!=CUDA_COUNT){
        printf("supposed devices number: %2d\n", CUDA_COUNT);
        exit(0);
    }
    
    int * couplings;
    int* spins[CUDA_COUNT], * shadow_spins[CUDA_COUNT];

    couplings       = (int*)malloc(N * N * sizeof(int));
    for(int i = 0; i < CUDA_COUNT; i++){
        spins[i]    = (int*)malloc(N * sizeof(int));
    }
    

    unsigned int* randnum_buf[CUDA_COUNT];
    
    int* spins_buf[CUDA_COUNT], * shadow_spins_buf[CUDA_COUNT];
    int* couplings_buf[CUDA_COUNT];
    
    for(int i = 0; i < CUDA_COUNT; i++){
        gpuErrchk(cudaSetDevice(i))
        gpuErrchk(cudaMalloc(&(couplings_buf[i]), N * (N / CUDA_COUNT) * sizeof(int)));
        gpuErrchk(cudaMalloc(&(spins_buf[i]), N * sizeof(int)));
        gpuErrchk(cudaMalloc(&(shadow_spins_buf[i]), N * sizeof(int)));
        gpuErrchk(cudaMalloc(&(randnum_buf[i]), N / CUDA_COUNT * sizeof(unsigned int)));
        for(int j = 1; j < CUDA_COUNT-1; i++){
            gpuErrchk(cudaDeviceEnablePeerAccess((i+j)%CUDA_COUNT, 0));
        }
    }

    // Read couplings file 
    FILE* instance = fopen("WK2000_1.rud", "r");
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

    printf("file input done.\n");
    fflush(stdout);

    // copy couplings coefficients
    for(int i = 0; i < CUDA_COUNT; i++){
        gpuErrchk(cudaSetDevice(i))
        gpuErrchk(cudaMemcpy(couplings_buf[i], couplings + i * N * (N / CUDA_COUNT), N * (N / CUDA_COUNT) * sizeof(int), cudaMemcpyHostToDevice));
    }
    printf("couplings into 2 GPU success.\n");
    fflush(stdout);

    // launching kernel
    dim3 grid(N / THREAD_NUM / CUDA_COUNT), block(THREAD_NUM);
    int offset[CUDA_COUNT];
    for(int i = 0; i < CUDA_COUNT; i++){
        gpuErrchk(cudaSetDevice(i));
        offset[i] = GRID_SIZE * i;
        randnum_init << <grid, block >> > (randnum_buf[i], offset[i]);
    }
    printf("randnum init in GPU success.\n");
    fflush(stdout);

    for (int i = 0; i < TIMES; i++) {
        clock_t begin = clock();
        for(int k = 0; k < CUDA_COUNT; k++){ //diff. devices
            gpuErrchk(cudaSetDevice(k));
            printf("k = %d\n", k);
            fflush(stdout);
            spins_init << <grid, block >> > (spins_buf, shadow_spins_buf, randnum_buf[k], k, offset[k]);
        }
        clock_t end = clock();
        printf("Spin init time %3d : %Lf\n", i, (double)(end - begin) / CLOCKS_PER_SEC);
        fflush(stdout);

        for(int k = 0; k < CUDA_COUNT; k++){
            gpuErrchk(cudaSetDevice(k));
            printf("k = %d\n", k);
            fflush(stdout);
            gpuErrchk(cudaMemcpy(*(spins+k), *(spins_buf+k), N * sizeof(int), cudaMemcpyDeviceToHost));
        }
        bool flg = false;
        for(int j = 0; j < N; j++) {
            for(int k = 0; k < CUDA_COUNT; k++){
                if(spins[k][j]!=spins[(k+1)%CUDA_COUNT][j]){
                    flg = true;
                    break;
                }
            }
            if(flg) break;
        }
        if(flg) printf("Spin chk fail.\n\n");
        else printf("Spin chk pass.\n\n");
        fflush(stdout);
    }

    // Release Objects
    free(couplings);
    for(int i = 0; i < CUDA_COUNT; i++){
        cudaFree(couplings_buf[i]);
    }
    return 0;
}
