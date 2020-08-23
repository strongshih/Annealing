#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda_profiler_api.h"

#define N 2048
#define THREADS 64
#define TIMES 10

#define M 16  // trotter layers
#define STEP 100

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert (cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ uint xorshift32 (uint *state)
{
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

__global__ void preapare_spins (char *spins, uint *randvals)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint rand = randvals[idx];

    // intializing spins
    #pragma unroll
    for (int m = 0; m < M; m++)
        spins[idx*M+m] = (char)(((xorshift32(&rand) & 1) << 1) - 1);
    
    randvals[idx] = rand;
}

__global__ void preapare_sigmas (char *spins, int *sigmas, char *couplings)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // intializing sigmas
    for (int m = 0; m < M; m++) {
        sigmas[idx*M+m] = 0;
    }

    #pragma unroll 16
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int m = 0; m < M; m++) {
            sigmas[idx*M+m] += spins[i*M+m]*couplings[idx*N+i];
        }
    }
}

__global__ void update_sigmas (int which_spin, 
                               int which_layer, 
                               char s, 
                               int *sigmas, 
                               char *couplings_buf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sigmas[idx*M+which_layer] = sigmas[idx*M+which_layer] - 2*s*couplings_buf[which_spin*N+idx];
}

void usage () 
{
    printf("Usage:\n");
    printf("       ./sqa [spin configuration]\n");
    exit(0);
}

int main (int argc, char *argv[]) 
{
    if (argc != 2) 
        usage();

    // initialize couplings
    char *couplings, *couplings_buf;
    couplings = (char*)malloc(N*N*sizeof(char));
    memset(couplings, '\0', N*N*sizeof(char));
    gpuErrchk( cudaMalloc(&couplings_buf, N*N*sizeof(char)) );

    // Read couplings file 
    FILE *instance = fopen(argv[1], "r");
    assert(instance != NULL);
    int a, b, w;
    fscanf(instance, "%d", &a);
    while (!feof(instance)) {
        fscanf(instance, "%d%d%d", &a, &b, &w);
        assert(a != b); // not dealing with external field
        couplings[a * N + b] = w;
        couplings[b * N + a] = w;
    }
    fclose(instance);

    // copy couplings to target device
    gpuErrchk( cudaMemcpy(couplings_buf, couplings, N*N*sizeof(char), cudaMemcpyHostToDevice) );

    // random number generation
    uint *randvals, *initRand;
    gpuErrchk( cudaMalloc(&randvals, N * sizeof(uint)) );
    initRand = (uint*)malloc(N*sizeof(uint));
    for (int i = 0; i < N; i++)
        initRand[i] = i;
    gpuErrchk( cudaMemcpy(randvals, initRand, N*sizeof(uint), cudaMemcpyHostToDevice) );

    // initialize spins
    char *s, *s_buf;
    s = (char*)malloc(M*N*sizeof(char));
    gpuErrchk( cudaMalloc(&s_buf, M*N*sizeof(char)) );
    
    // initialize 
    int *sigma, *sigma_buf;
    sigma = (int*)malloc(M*N*sizeof(int));
    gpuErrchk( cudaMalloc(&sigma_buf, M*N*sizeof(int)) );

    // launching kernel
    dim3 grid(N/THREADS), block(THREADS);
    int results[TIMES] = {0};
    int delta;
    float increase = (8 - 1/(float)16) / (float)STEP;
    float G0 = 8.;
    
    for (int t = 0; t < TIMES; t++) {
        float beta = 1/(float)16;
        
        preapare_spins<<<grid, block>>>(s_buf, randvals);
        preapare_sigmas<<<grid, block>>>(s_buf, sigma_buf, couplings_buf);
        gpuErrchk( cudaMemcpy(s, s_buf, M*N*sizeof(char), cudaMemcpyDeviceToHost) );

        double curr = 0.;

        for (int p = 0; p < STEP; p++) {
            float Gamma = G0*(1.-(float)p/(float)STEP);
            float J_perp = -0.5*log(tanh((Gamma/M)*beta))/beta;
            clock_t begin = clock();
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    int idx = n*M+m;
                    gpuErrchk( cudaMemcpy(&delta, sigma_buf+idx, 1*sizeof(int), cudaMemcpyDeviceToHost) );
                    int upper = (m == 0 ? M-1 : m-1);
                    int lower = (m == m-1 ? 0 : m+1);
                    delta = 2*M*s[idx]*(delta - M*J_perp*(s[n*M+upper] + s[n*M+lower]));
                    if ( (-log(rand() / (float) RAND_MAX) / beta) > delta ) {
                        update_sigmas<<<grid, block>>>(n, m, s[idx], sigma_buf, couplings_buf);
                        s[idx] = -s[idx];
                    }
                }
            }
            beta += increase;
            clock_t end = clock();
            double duration = (double)(end-begin) / CLOCKS_PER_SEC;
            curr += duration;
            
            int E = 0;
            for (int i = 0; i < N; i++)
                for (int j = i+1; j < N; j++)
                    E += -s[i*M+0]*s[j*M+0]*couplings[i*N+j];
            results[t] = E;
            printf("curr: %10lf, energy: %10d\n", curr, E);
        }
        
    }

    // Write statistics to file
    FILE *output;
    output = fopen("output.txt", "w");
    for (int i = 0; i < TIMES; i++)
         fprintf(output, "%d\n", results[i]);
    fclose(output);

    // Release Objects
    free(couplings);
    free(initRand);
    free(s);
    free(sigma);
    cudaFree(couplings_buf);
    cudaFree(s_buf);
    cudaFree(sigma_buf);
    cudaFree(randvals);
    return 0;
}