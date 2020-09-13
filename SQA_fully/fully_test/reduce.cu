#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda_profiler_api.h"

#define N 32768
#define M 32
#define THREADS 1024
#define TIMES 1

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert (cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    #pragma unroll
	for (int m = 0; m < M; m++) {
        sdata[tid] = g_idata[i + m*N] + g_idata[i+blockDim.x + m*N];
    	
		__syncthreads();
    
		if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
		if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
		if (tid < 32) warpReduce<blockSize>(sdata, tid);
		if (tid == 0) g_odata[blockIdx.x + m*(N/THREADS/2)] = sdata[0];
	}
}

int main() {
    int *a, *a_buf, *out, *out_buf;
    a = (int*)malloc(sizeof(int)*M*N);
    out = (int*)malloc(sizeof(int)*M*N);
	for (int m = 0; m < M; m++) {
		for (int i = 0; i < N; i++) {
			a[m*N+i] = i+100*m;
		}
	}
    gpuErrchk( cudaMalloc(&a_buf, M*N*sizeof(int)) );
    gpuErrchk( cudaMemcpy(a_buf, a, M*N*sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMalloc(&out_buf, M*N*sizeof(int)) );

    dim3 grid(N/THREADS/2), block(THREADS);
    reduce<THREADS><<<grid, block, THREADS*sizeof(int)>>>(a_buf, out_buf);

    gpuErrchk( cudaMemcpy(out, out_buf, M*N*sizeof(int), cudaMemcpyDeviceToHost) );

	for (int m = 0; m < M; m++) {
		int sum = 0;
		for (int i = 0; i < N/THREADS/2; i++) {
			sum += out[m*(N/THREADS/2)+i];
		}
		int compare = 0;
		for (int i = 0; i < N; i++) {
			compare += (i+100*m);
		}
		printf("%d %d\n", sum, compare);
	}
}
