#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
 
#define N 4096
#define EDGE 64
#define THREADS 16
#define TIMES 10
#define SWEEP 500
#define MAX 4294967295.0

// parameter settings
#define K 1
#define detune 0.5
#define xi 0.1
#define epsilon 0.01
#define deltaT 1

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert (cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef struct point {
	float x;
	float y;
} Point;


__device__ uint xorshift32 (uint *state)
{
	uint x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return x;
}

__global__ void prepare_points (Point *p,
                        	    uint *randvals) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    p[idx].x = ((xorshift32(&randvals[idx]) / (float) MAX) / 10.0);
    p[idx].y = ((xorshift32(&randvals[idx]) / (float) MAX) / 10.0);
}

__global__ void Bifurcation (Point *p,
							 int* couplings, 
							 float amplitude)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int oldX = p[idx].x;
    int oldY = p[idx].y;

    // update x coordinate
    int newX = detune*oldY*deltaT + oldX;
    p[idx].x = newX;

    // update y coordinate
    int newY = -K * (oldX * oldX * oldX)*deltaT;
    int s = 0;
    for (int i = 0; i < N; i++)
    	s += couplings[idx*N+i]*p[idx].x;
    s *= (xi*deltaT);
    newY += s;
    newY += oldY;
    p[idx].y = newY;
}

void usage () 
{
	printf("Usage:\n");
	printf("       ./Bifurcation-cuda [spin configuration]\n");
	exit(0);
}

int main (int argc, char *argv[]) 
{
	if (argc != 2) 
		usage();

	// initialize couplings
	int *couplings, *couplings_buf;
    couplings = (int*)malloc(N*N*sizeof(int));
	memset(couplings, '\0', N*N*sizeof(int));
	gpuErrchk( cudaMalloc(&couplings_buf, N*N*sizeof(int)) );

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
	gpuErrchk( cudaMemcpy(couplings_buf, couplings, N*N*sizeof(int), cudaMemcpyHostToDevice) );

	// initialize random number
	uint *randvals, *initRand;
	gpuErrchk( cudaMalloc(&randvals, N * sizeof(uint)) );
    initRand = (uint*)malloc(N*sizeof(uint));
	for (int i = 0; i < N; i++)
		initRand[i] = i;
	gpuErrchk( cudaMemcpy(randvals, initRand, N*sizeof(uint), cudaMemcpyHostToDevice) );

	// initialize points
	Point *points, *points_buf;
    points = (Point*)malloc(N*sizeof(Point));
	gpuErrchk( cudaMalloc(&points_buf, N*sizeof(Point)) );

	// launching kernel
	dim3 grid(N), block(1);
	int results[TIMES] = {0};
	for (int x = 0; x < TIMES; x++) {
		prepare_points<<<grid, block>>>(points_buf, randvals);
		for (int s = 0; s < SWEEP; s++) {
			float amplitude = epsilon*s;
			Bifurcation<<<grid, block>>>(points_buf, couplings_buf, 
										 amplitude);
		}
		// Get Result from device
		gpuErrchk( cudaMemcpy(points, points_buf, N*sizeof(Point), cudaMemcpyDeviceToHost) );

		// calculate energy
		int E = 0;
		for (int i = 0; i < N; i++) {
			for (int j = i+1; j < N; j++) {
				int a = points[i].x > 0.0 ? 1 : -1;
				int b = points[j].x > 0.0 ? 1 : -1;
				E += a*b*couplings[i*N+j];
			}
		}
		results[x] = -E;
	}

	// Write statistics to file
	FILE *output;
	output = fopen("output.txt", "w");
	for (int i = 0; i < TIMES; i++)
 		fprintf(output, "%d\n", results[i]);
	fclose(output);

	// Release Objects
	free(couplings);
	free(points);
	free(initRand);
	cudaFree(couplings_buf);
	cudaFree(points_buf);
	cudaFree(randvals);
	return 0;
}
