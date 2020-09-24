#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda_profiler_api.h"

#define N 4096
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

__global__ void preapare_spins (char *spins, uint *randvals)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint rand = randvals[idx];

    // intializing spins
    #pragma unroll
    for (int m = 0; m < M; m++) {
        char temp = (char)(((xorshift32(&rand) & 1) << 1) - 1);
        spins[idx*M+m] = temp;
	}
    randvals[idx] = rand;
}

__global__ void update_spins (int iter, 
                              char *spins, 
                              char *couplings_buf,
                              float J_perp,
                              float beta,
                              uint *randvals,
                              int *out)
{
    int m = threadIdx.x;
	 
	int target_spin = (iter - m + N) & (N-1);
    int target_pos = target_spin*M + m;

	float delta = 0.;
	
	for (int t = (N/THREADS/2)*m; t < (N/THREADS/2)*(m+1); t++)
		delta += (float)out[t];
	
	uint rand = randvals[target_spin];
	
	int upper = (m == 0 ? M-1 : m-1);
	int lower = (m == M-1 ? 0 : m+1);
	
	delta = 2.*(float)M*(float)spins[target_pos]*
			(delta - (float)M*J_perp*(float)(spins[target_spin*M+upper] + spins[target_spin*M+lower]));
	
	if ( (-log((float)rand / (float) MAX) / beta) > delta ) {
		spins[target_pos] = -spins[target_pos];
	}
	
	randvals[target_spin] = rand;
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
	for (int m = 0; m < M; m++) {
		if (blockSize >= 64) sdata[m*THREADS + tid] += sdata[m*THREADS + tid + 32];
		if (blockSize >= 32) sdata[m*THREADS + tid] += sdata[m*THREADS + tid + 16];
		if (blockSize >= 16) sdata[m*THREADS + tid] += sdata[m*THREADS + tid + 8];
		if (blockSize >= 8)  sdata[m*THREADS + tid] += sdata[m*THREADS + tid + 4];
		if (blockSize >= 4)  sdata[m*THREADS + tid] += sdata[m*THREADS + tid + 2];
		if (blockSize >= 2)  sdata[m*THREADS + tid] += sdata[m*THREADS + tid + 1];
	}
}

template <unsigned int blockSize>
__global__ void reduce(int iter, char *g_idata, int *g_odata, char *couplings) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    #pragma unroll
	for (int m = 0; m < M; m++) {
		int target_spin = (iter - m + N) & (N-1);
		sdata[m*THREADS+tid] = g_idata[i*M + m]*couplings[target_spin*N+i] + \
					 g_idata[(i+blockDim.x)*M + m]*couplings[target_spin*N+i+blockDim.x];
	}
	__syncthreads();

	if (blockSize >= 1024) { 
		if (tid < 512) { 
			#pragma unroll
			for (int m = 0; m < M; m++) 
				sdata[m*THREADS+tid] += sdata[m*THREADS + tid + 512]; 
		} 
		__syncthreads(); 
	}
	if (blockSize >= 512) { 
		if (tid < 256) { 
			#pragma unroll
			for (int m = 0; m < M; m++) 
				sdata[m*THREADS+tid] += sdata[m*THREADS + tid + 256]; 
		} 
		__syncthreads(); 
	}
	if (blockSize >= 256) { 
		if (tid < 128) { 
			#pragma unroll
			for (int m = 0; m < M; m++) 
				sdata[m*THREADS+tid] += sdata[m*THREADS + tid + 128]; 
		} 
		__syncthreads(); 
	}
	if (blockSize >= 128) { 
		if (tid < 64) { 
			#pragma unroll
			for (int m = 0; m < M; m++) 
				sdata[m*THREADS+tid] += sdata[m*THREADS + tid + 64]; 
		} 
		__syncthreads(); 
	}
	if (tid < 32) warpReduce<blockSize>(sdata, tid);

	// put pack
	for (int m = 0; m < M; m++)
		if (tid == 0) g_odata[blockIdx.x + m*(N/THREADS/2)] = sdata[m*THREADS];
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
    char *s_buf, *s;
    s = (char*)malloc(M*N*sizeof(char));
    gpuErrchk( cudaMalloc(&s_buf, M*N*sizeof(char)) );

    // out
    int *out_buf;
    gpuErrchk( cudaMalloc(&out_buf, M*(N/THREADS/2)*sizeof(int)) );

    // launching kernel
    dim3 grid(N/THREADS/2), block(THREADS);
	dim3 grid2(N/THREADS);
    int results[TIMES] = {0};
    float increase = (8 - 1/(float)16) / (float)STEP;
    float G0 = 8.;
    
    for (int t = 0; t < TIMES; t++) {
        float beta = 1/(float)16;
        
        // initialize spins and sigmas
        preapare_spins<<<grid2, block>>>(s_buf, randvals);

        double curr = 0.;

        for (int p = 0; p < STEP; p++) {
            float Gamma = G0*(1.-(float)p/(float)STEP);
            float J_perp = -0.5*log(tanh((Gamma/M)*beta))/beta;
            clock_t begin = clock();
            for (int a = 0; a < M+N-1; a++) {
				reduce<THREADS><<<grid, block, M*THREADS*sizeof(int)>>>(a, s_buf, out_buf, couplings_buf);
				update_spins<<<1, M>>>(a, s_buf, couplings_buf, J_perp, beta, randvals, out_buf);
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
            printf("curr: %10lf, energy: %10d\n", curr, E);
        }
        // printf("Per time step: %10lf, M: %10d, N: %10d\n", curr/(float)STEP, M, N);
        
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
    cudaFree(randvals);
    return 0;
}
