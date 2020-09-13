#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda_profiler_api.h"

#define N 16384
#define THREADS 64
#define TIMES 1

#define MAX 4294967295.0
#define STEP 100
#define M 64

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

__global__ void preapare_spins (char *spins, char *spins_out, uint *randvals, uint *randvals2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint rand = randvals[idx];

    // intializing spins
    #pragma unroll
    for (int m = 0; m < M; m++) {
        char temp = (char)(((xorshift32(&rand) & 1) << 1) - 1);
        spins[idx*M+m] = temp;
        spins_out[idx*M+m] = temp;
	}
    randvals[idx] = rand;
	randvals2[idx] = rand;
}

__global__ void preapare_sigmas (char *spins, int *sigmas, int *sigmas_out, char *couplings)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // intializing sigmas
    for (int m = 0; m < M; m++) {
        sigmas[idx*M+m] = 0;
        sigmas_out[idx*M+m] = 0;
    }

    #pragma unroll 16
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int m = 0; m < M; m++) {
            sigmas[idx*M+m] += spins[i*M+m]*couplings[idx*N+i];
        }
        #pragma unroll
		for (int m = 0; m < M; m++) {
		    sigmas_out[idx*M+m] = sigmas[idx*M+m];
		}
    }
}

__global__ void update_sigmas (int iter, 
                               char *spins, 
                               int *sigmas, 
                               char *couplings_buf,
                               float J_perp,
                               float beta,
                               uint *randvals,
							   char *spins_out,
							   int *sigmas_out,
							   uint *randvals_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int m = 0; m < M; m++) { 
    	int target_spin = iter - m;
        int target_pos = target_spin*M+m;
        int which_layer = m;
        int which_spin = i;
		int idx = i*M+m;

		if (target_spin >= 0 && target_spin < N) {
	        uint rand = randvals[target_spin];
			int upper = (which_layer == 0 ? M-1 : which_layer-1);
			int lower = (which_layer == M-1 ? 0 : which_layer+1);
			float delta = 2.*(float)M*(float)spins[target_pos]*
					((float)sigmas[target_pos] - (float)M*J_perp*(float)(spins[target_spin*M+upper] + spins[target_spin*M+lower]));
			if ( (-log((float)rand / (float) MAX) / beta) > delta ) {
				sigmas_out[idx] = sigmas[idx] - 2*spins[target_pos]*couplings_buf[target_spin*N+which_spin];
				if (idx == target_pos) {  
					spins_out[idx] = -spins[idx];
				} else {
					spins_out[idx] = spins[idx];
				}
			} else {
				sigmas_out[idx] = sigmas[idx];
				spins_out[idx] = spins[idx];
			}
	        randvals_out[target_spin] = rand;
		}
    }
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
    uint *randvals2;
    gpuErrchk( cudaMalloc(&randvals2, N * sizeof(uint)) );

    // initialize spins
    char *s_buf, *s;
    s = (char*)malloc(M*N*sizeof(char));
    gpuErrchk( cudaMalloc(&s_buf, M*N*sizeof(char)) );
    char *s_out_buf;
    gpuErrchk( cudaMalloc(&s_out_buf, M*N*sizeof(char)) );
    
    // initialize 
    int *sigma_buf;
    gpuErrchk( cudaMalloc(&sigma_buf, M*N*sizeof(int)) );
    int *sigma_out_buf;
    gpuErrchk( cudaMalloc(&sigma_out_buf, M*N*sizeof(int)) );

    // launching kernel
    dim3 grid(N/THREADS), block(THREADS);
    int results[TIMES] = {0};
    float increase = (8 - 1/(float)16) / (float)STEP;
    float G0 = 8.;
    
    for (int t = 0; t < TIMES; t++) {
        float beta = 1/(float)16;
        
        // initialize spins and sigmas
        preapare_spins<<<grid, block>>>(s_buf, s_out_buf, randvals, randvals2);
        preapare_sigmas<<<grid, block>>>(s_buf, sigma_buf, sigma_out_buf, couplings_buf);

        double curr = 0.;

        for (int p = 0; p < STEP; p++) {
            float Gamma = G0*(1.-(float)p/(float)STEP);
            float J_perp = -0.5*log(tanh((Gamma/M)*beta))/beta;
            clock_t begin = clock();
            for (int a = 0; a < M+N-1; a++) {
				if (a % 2 == 0) {
					update_sigmas<<<grid, block>>>(a, s_buf, sigma_buf,
									   couplings_buf, J_perp, beta, randvals, s_out_buf, sigma_out_buf, randvals2);
				} else {
					update_sigmas<<<grid, block>>>(a, s_out_buf, sigma_out_buf,
									   couplings_buf, J_perp, beta, randvals2, s_buf, sigma_buf, randvals);
				}
            }
            beta += increase;
            clock_t end = clock();
            double duration = (double)(end-begin) / CLOCKS_PER_SEC;
            curr += duration;

            gpuErrchk( cudaMemcpy(s, s_buf, M*N*sizeof(char), cudaMemcpyDeviceToHost) );
            int E = 0;
            for (int i = 0; i < N; i++)
                for (int j = i+1; j < N; j++)
                    E += -s[i*M+0]*s[j*M+0]*couplings[i*N+j];
            results[t] = E;
            // printf("curr: %10lf, energy: %10d\n", curr, E);
        }
        printf("Per time step: %10lf, M: %10d, N: %10d\n", curr/(float)STEP, M, N);
        
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
    cudaFree(couplings_buf);
    cudaFree(s_buf);
    cudaFree(sigma_buf);
    cudaFree(randvals);
    return 0;
}
