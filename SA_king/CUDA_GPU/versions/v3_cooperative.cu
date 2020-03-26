#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cooperative_groups.h>
#include <cuda.h>
 
#define N 16384
#define EDGE 128
#define THREADS 16
#define TIMES 10
#define SWEEP 1000
#define MAX 4294967295.0

using namespace cooperative_groups;

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);            \
      return EXIT_FAILURE;}} while(0)

__device__ uint xorshift32(uint *state)
{
	uint x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return x;
}

__global__ void prepare_spins(int *spins, 
							  uint *randvals) 
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	spins[idx] = ((xorshift32(&randvals[idx]) & 1) << 1) - 1;
}

__device__ void update(int* spins, 
					  const int* couplings, 
					  int signal, 
					  float beta,
					  uint *randvals)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	// annealing (calculate 8 neighbor energy difference)
	int n = idx-EDGE-1;
	int e0 = ((idx%EDGE) == 0 || idx < EDGE) ?
				0 : couplings[idx*9+0]*spins[n];  

	n = idx-EDGE;
	int e1 = (idx < EDGE) ?
				0 : couplings[idx*9+1]*spins[n];  

	n = idx-EDGE+1;
	int e2 = (((idx+1)%EDGE) == 0 || idx < EDGE) ?
				0 : couplings[idx*9+2]*spins[n];  

	n = idx-1;
	int e3 = ((idx%EDGE) == 0) ?
				0 : couplings[idx*9+3]*spins[n];  

	n = idx+1;
	int e4 = (((idx+1)%EDGE) == 0) ?
				0 : couplings[idx*9+4]*spins[n];  

	n = idx+EDGE-1;
	int e5 = ((idx%EDGE) == 0 || idx >= EDGE*(EDGE-1)) ?
				0 : couplings[idx*9+5]*spins[n];  

	n = idx+EDGE;
	int e6 = (idx >= EDGE*(EDGE-1)) ?
				0 : couplings[idx*9+6]*spins[n];  

	n = idx+EDGE+1;
	int e7 = (((idx+1)%EDGE) == 0 || idx >= EDGE*(EDGE-1)) ?
				0 : couplings[idx*9+7]*spins[n];  

	int difference = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7 + couplings[idx*9+8];
	difference = -2*difference*spins[idx]*beta;
	int row = (idx/EDGE)%2;
	int col = (idx%EDGE)%2;
	int group = 2*row + col;
	if (signal == group && difference > log(xorshift32(&randvals[idx]) / MAX)) {
		spins[idx] = -spins[idx];
	}
}

__global__ void ising(int* spins, 
					  const int* couplings, 
					  uint *randvals)
{
	grid_group g = this_grid();
	float beta = 0.1;
	float increase = (3.0 - 0.1) / (float) SWEEP;
	for (int s = 0; s < SWEEP; s++) {
		for (int i = 0; i < 4; i++) {
			update(spins, couplings, i, beta, randvals);
			g.sync();
		}
		beta += increase;
	}
}

void usage () 
{
	printf("Usage:\n");
	printf("       ./Ising-opencl [spin configuration]\n");
	exit(0);
}

int relation (int a, int b) 
{
	switch (b-a) {
		case 0:
			return 8;
		case -EDGE-1:
			return 0;
		case -EDGE:
			return 1;
		case -EDGE+1:
			return 2;
		case -1:
			return 3;
		case 1:
			return 4;
		case EDGE-1:
			return 5;
		case EDGE:
			return 6;
		case EDGE+1:
			return 7;
		default:
			return -1;
	}
}

int main (int argc, char *argv[]) 
{
	if (argc != 2) 
		usage();

	// initialize parameters
	int *couplings, *couplings_buf, *spins_buf, *spins;
    couplings = (int*)malloc(N*9*sizeof(int));
    spins = (int*)malloc(N*sizeof(int));
	memset(couplings, '\0', N*9*sizeof(int));
	CUDA_CALL( cudaMalloc(&couplings_buf, N*9*sizeof(int)) );
	CUDA_CALL( cudaMalloc(&spins_buf, N*sizeof(int)) );

	// Read couplings file 
	FILE *instance = fopen(argv[1], "r");
	assert(instance != NULL);
	int a, b, w;
	fscanf(instance, "%d", &a);
	while (!feof(instance)) {
		fscanf(instance, "%d%d%d", &a, &b, &w);
		int r = relation(a, b);
		if (r == -1) {
			assert(false);
		} else {
			couplings[9*a+r] = w;
			r = relation(b, a);
			couplings[9*b+r] = w;
		}
	}
	fclose(instance);

	// copy couplings coefficients (when N is large, take lots of time)
	CUDA_CALL( cudaMemcpy(couplings_buf, couplings, N*9*sizeof(int), cudaMemcpyHostToDevice) );
	printf("Finish copying coefficients\n");

	// random number generation
	uint *randvals, *initRand;
	CUDA_CALL( cudaMalloc(&randvals, N * sizeof(uint)) );
    initRand = (uint*)malloc(N*sizeof(uint));
	for (int i = 0; i < N; i++)
		initRand[i] = i;
	CUDA_CALL( cudaMemcpy(randvals, initRand, N*sizeof(uint), cudaMemcpyHostToDevice) );
	printf("Finish intialize random number\n");

	// work group division
	dim3 grid(EDGE/THREADS, EDGE/THREADS), block(THREADS, THREADS);
	void *kernelArgs[] = {
		(void *)&spins_buf, (void *)&couplings_buf, 
		(void *)&randvals,
	};

	int results[TIMES] = {0};
	for (int x = 0; x < TIMES; x++) {
		prepare_spins<<<grid, block>>>(spins_buf, randvals);
		//ising<<<grid, block>>>(spins_buf, couplings_buf, randvals);
		cudaLaunchCooperativeKernel((void*)ising, 
				grid, block, kernelArgs, 1024*sizeof(float), NULL);

		// Get result from device
		CUDA_CALL( cudaMemcpy(spins, spins_buf, N*sizeof(int), cudaMemcpyDeviceToHost) );
		// CUDA_CALL( cudaPeekAtLastError() );
		int E = 0;
		for (int i = 0; i < N; i++) {
			E += spins[i] * couplings[9*i+8];
			for (int j = i+1; j < N; j++) {
				int r = relation(i, j);
				if (r == -1) continue;
				E += spins[i] * spins[j] * couplings[9*i+r];
			}
		}
		results[x] = -E;
	}

	// Write results to file
	FILE *output;
	output = fopen("output.txt", "w");
	for (int i = 0; i < TIMES; i++)
 		fprintf(output, "%d\n", results[i]);
	fclose(output);

	// Release Objects
	free(couplings);
	cudaFree(spins_buf);
	cudaFree(couplings_buf);
	return 0;
}
