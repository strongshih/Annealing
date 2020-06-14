#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cuda_profiler_api.h"
#include "cuda_fp16.h"
 
#define N 32768ull
#define THREADS 64
#define TIMES 1

// parameter settings (p > detune - xi, bifurcation point)
#define STEP 20
#define M 2
#define MAX 4294967295.0

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

__global__ void prepare_points (half *x,
                                half *y,
                                uint *randvals) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    x[idx] = ((half)(xorshift32(&randvals[idx]) / MAX / 10.0));
    y[idx] = ((half)(xorshift32(&randvals[idx]) / MAX / 10.0));
}

__global__ void UpdateTwice (half *x,
                             half *y,
				             half amplitude,
							 half detune,
							 half deltaT,
							 half K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	half lx = x[idx];
	half ly = y[idx];

#pragma unroll
	for (int i = 0; i < M; i++) {
        lx = lx + detune*ly*deltaT;
        ly = ly-deltaT*(K*lx*lx*lx + (detune - amplitude)*lx);
	}

	x[idx] = lx;
	y[idx] = ly;
}

void print_energy(half *x, half *x_buf, half *couplings, double curr, int t)
{
	gpuErrchk( cudaMemcpy(x, x_buf, N*sizeof(half), cudaMemcpyDeviceToHost) );
	int E = 0;
	for (int i = 0; i < N; i++) {
		for (int j = i+1; j < N; j++) {
			int a = (float) x[i] > 0.0 ? 1 : -1;
			int b = (float) x[j] > 0.0 ? 1 : -1;
			E += a*b*((int)(float)couplings[i*N+j]);
		}
	}
	printf("%d %lf %d\n", t, curr, -E);
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

	half K = 1.;
    half detune = 1.;
    half xi = 0.004736; // 0.7*detune / (rho * sqrt(N))
    // half xi = 0.01565; // 0.7*detune / (rho * sqrt(N))
    half deltaT = 0.5;
    half DeltaT = 1.0;

    // initialize couplings
    half *couplings, *couplings_buf;
    couplings = (half*)malloc(N*N*sizeof(half));
    memset(couplings, '\0', N*N*sizeof(half));
    gpuErrchk( cudaMalloc(&couplings_buf, N*N*sizeof(half)) );

    // Read couplings file 
    FILE *instance = fopen(argv[1], "r");
    assert(instance != NULL);
    int a, b, w;
    fscanf(instance, "%d", &a);
    while (!feof(instance)) {
        fscanf(instance, "%d%d%d", &a, &b, &w);
        assert(a != b); // not dealing with external field
        couplings[a * N + b] = (half)((float)w);
        couplings[b * N + a] = (half)((float)w);
    }
    fclose(instance);

	// cublas
	cublasHandle_t handle;
	cublasCreate(&handle);

    // copy couplings to target device
    gpuErrchk( cudaMemcpy(couplings_buf, couplings, N*N*sizeof(half), cudaMemcpyHostToDevice) );

    // initialize random number
    uint *randvals, *initRand;
    gpuErrchk( cudaMalloc(&randvals, N * sizeof(uint)) );
    initRand = (uint*)malloc(N*sizeof(uint));
    for (int i = 0; i < N; i++)
        initRand[i] = i;
    gpuErrchk( cudaMemcpy(randvals, initRand, N*sizeof(uint), cudaMemcpyHostToDevice) );

    // initialize points
    half *x, *x_buf, *y, *y_buf;
    x = (half*)malloc(N*sizeof(half));
    gpuErrchk( cudaMalloc(&x_buf, N*sizeof(half)) );
    y = (half*)malloc(N*sizeof(half));
    gpuErrchk( cudaMalloc(&y_buf, N*sizeof(half)) );

    // launching kernel
	half coeff1 = 0.01565; // DeltaT * xi;
	half coeff2 = 1.;
    dim3 grid(N/THREADS), block(THREADS);
    int results[TIMES] = {0};
	
	cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
	//cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    for (int t = 0; t < TIMES; t++) {
        prepare_points<<<grid, block>>>(x_buf, y_buf, randvals);
		double curr = 0.;
        for (int s = 0; s < STEP; s++) {
			float temp_amp = (float)s/(float)STEP;
            half amplitude = (half)(temp_amp);
			clock_t begin = clock();
			UpdateTwice<<<grid, block>>>(x_buf, y_buf, amplitude, detune, deltaT, K);
			cublasHgemm(handle, 
						CUBLAS_OP_N, 
						CUBLAS_OP_N,
						N, 1, N,
						&coeff1,
						couplings_buf, N, 
						x_buf, N,
						&coeff2, 
						y_buf, N);
			clock_t end = clock();
			double duration = (double)(end-begin) / CLOCKS_PER_SEC;
			curr += duration;
			printf("%f\n", curr);
            print_energy(x, x_buf, couplings, curr, t);
        }
        // Get Result from device
        gpuErrchk( cudaMemcpy(x, x_buf, N*sizeof(half), cudaMemcpyDeviceToHost) );

        // calculate energy
        int E = 0;
        for (int i = 0; i < N; i++) {
            for (int j = i+1; j < N; j++) {
				int a = (float) x[i] > 0.0 ? 1 : -1;
				int b = (float) x[j] > 0.0 ? 1 : -1;
                E += a*b*((int)(float)couplings[i*N+j]);
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
