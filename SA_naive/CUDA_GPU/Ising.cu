#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
 
#define MAXDEVICE 10
#define MAXK 2048
#define N 512
#define TIMES 1024
#define NANO2SECOND 1000000000.0

#define SWEEP 200
#define MAX 4294967295

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
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

__global__ void ising(int* couplings, int* results)
{
	// random number
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint randnum = idx + 1337;
	float beta = 0.1; // from 0.1 to 3.0
	float increase = (3.0 - 0.1) / (float) SWEEP;

	// spin initialization
	int spins[N];
	for (int i = 0; i < N; i++)
		spins[i] = ((xorshift32(&randnum) & 1) << 1) - 1; 

	// annealing
	for (int i = 0; i < SWEEP; i++) {
		beta += increase;
		float r = xorshift32(&randnum) / (float) MAX;
		for (int n = 0; n < N; n++) {
			int difference = 0;
			for (int j = 0; j < N; j++)
				difference += couplings[n*N+j]*spins[j];
			difference = -1 * difference * spins[n];
			if ((difference * beta) > log(r)) {
				spins[n] = -spins[n];
			}
		}
	}

	// calculate result
	int E = 0;
	for (int i = 0; i < N; i++)
		for (int j = i; j < N; j++)
			E += spins[i]*spins[j]*couplings[i*N+j];
	results[idx] = -E;
}

void usage() {
	printf("Usage:\n");
	printf("       ./Ising-opencl [spin configuration]\n");
	exit(0);
}

int main (int argc, char *argv[]) {
	if (argc != 2) 
		usage();

	// initialize parameters
	int *couplings, *results, *couplings_buf, *results_buf;
    couplings = (int*)malloc(N*N*sizeof(int));
	results = (int*)malloc(TIMES*sizeof(int));
	memset(couplings, '\0', N*N*sizeof(int));
	gpuErrchk( cudaMalloc(&couplings_buf, N*N*sizeof(int)) );
	gpuErrchk( cudaMalloc(&results_buf, TIMES*sizeof(int)) );

	// Read couplings file 
	FILE *instance = fopen(argv[1], "r");
	assert(instance != NULL);
	int a, b, w;
	fscanf(instance, "%d", &a);
	while (!feof(instance)) {
		fscanf(instance, "%d%d%d", &a, &b, &w);
		couplings[a * N + b] = w;
		couplings[b * N + a] = w;
	}
	fclose(instance);

	// copy couplings coefficients
	gpuErrchk( cudaMemcpy(couplings_buf, couplings, N*N*sizeof(int), cudaMemcpyHostToDevice) );

	// launching kernel
	dim3 grid(TIMES), block(1);
	ising<<<grid, block>>>(couplings_buf, results_buf);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// Get Result from device
	gpuErrchk( cudaMemcpy(results, results_buf, TIMES*sizeof(int), cudaMemcpyDeviceToHost) );

	// Write statistics to file
	FILE *output;
	output = fopen("output.txt", "w");
	for (int i = 0; i < TIMES; i++)
 		fprintf(output, "%d\n", results[i]);
	fclose(output);

	// Release Objects
	free(results);
	free(couplings);
	cudaFree(results_buf);
	cudaFree(couplings_buf);
	return 0;
}
