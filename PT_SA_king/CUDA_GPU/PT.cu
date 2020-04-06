#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda.h>

#define N 1048576
#define EDGE 1024
#define THREADS 32
#define TIMES 10
#define SWEEP 20
#define REPLICA 40
#define MAX 4294967295.0

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
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

__global__ void prepare_spins (int *spins, 
                               uint *randvals) 
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    for (int r = 0; r < REPLICA; r++)
        spins[r*N+idx] = ((xorshift32(&randvals[idx]) & 1) << 1) - 1;
}

__device__ int sumNeighbor (int *spins,
                            int *couplings,
                            int idx,
                            int replica) 
{
    int n = idx-EDGE-1;
    int e0 = ((idx%EDGE) == 0 || idx < EDGE) ?
                0 : couplings[idx*9+0]*spins[replica*N+n];  
    n = idx-EDGE;
    int e1 = (idx < EDGE) ?
                0 : couplings[idx*9+1]*spins[replica*N+n];  
    n = idx-EDGE+1;
    int e2 = (((idx+1)%EDGE) == 0 || idx < EDGE) ?
                0 : couplings[idx*9+2]*spins[replica*N+n];  
    n = idx-1;
    int e3 = ((idx%EDGE) == 0) ?
                0 : couplings[idx*9+3]*spins[replica*N+n];  
    n = idx+1;
    int e4 = (((idx+1)%EDGE) == 0) ?
                0 : couplings[idx*9+4]*spins[replica*N+n];  
    n = idx+EDGE-1;
    int e5 = ((idx%EDGE) == 0 || idx >= EDGE*(EDGE-1)) ?
                0 : couplings[idx*9+5]*spins[replica*N+n];  
    n = idx+EDGE;
    int e6 = (idx >= EDGE*(EDGE-1)) ?
                0 : couplings[idx*9+6]*spins[replica*N+n];  
    n = idx+EDGE+1;
    int e7 = (((idx+1)%EDGE) == 0 || idx >= EDGE*(EDGE-1)) ?
                0 : couplings[idx*9+7]*spins[replica*N+n];
    int sumN = (e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7);
    return sumN;
}

__global__ void ising (int* spins, 
                       int* couplings,
                       int signal, 
                       float beta,
                       uint *randvals,
                       int replica)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int difference = -2*beta*spins[replica*N+idx]*
                (sumNeighbor(spins, couplings, idx, replica)+couplings[idx*9+8]);
    int row = (idx/EDGE)%2;
    int col = (idx%EDGE)%2;
    int group = 2*row + col;
    if (signal == group && difference > log(xorshift32(&randvals[idx]) / MAX)) {
        spins[replica*N+idx] = -spins[replica*N+idx];
    }
}

__global__ void reduction (int *spins, 
                           int *couplings,
                           int replica,
                           int *sum_result)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = (threadIdx.y * blockDim.x) + threadIdx.x;
    int idx = blockId * (blockDim.x * blockDim.y) + tid;
    __shared__ int sdata[THREADS*THREADS];

    sdata[tid] = spins[replica*N+idx]*
            (sumNeighbor(spins, couplings, idx, replica)+2*couplings[idx*9+8]);
    __syncthreads();
    
    for (int s = (blockDim.x*blockDim.y)/2; s > 0; s >>= 1) { 
        if (tid < s) {
            sdata[tid] += sdata[tid + s]; 
        }
        __syncthreads(); 
    }

    if (tid == 0) {
        sum_result[blockId] = sdata[0];
    }
}

__global__ void swap (int* spins, 
                       int r)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int tmp = spins[r*N+idx];
    spins[r*N+idx] = spins[(r+1)*N+idx];
    spins[(r+1)*N+idx] = tmp;
}

int reduced_sum (int *spins, int *couplings, int r, int *sum, int *sum_buf) {
    dim3 grid(EDGE/THREADS, EDGE/THREADS), block(THREADS, THREADS);
    reduction<<<grid, block>>>(spins, couplings, r, sum_buf);
        CUDA_CALL( cudaPeekAtLastError() );
    CUDA_CALL( cudaMemcpy(sum, sum_buf, (EDGE/THREADS)*(EDGE/THREADS)*sizeof(int), cudaMemcpyDeviceToHost) );
    int ret = 0;
    for (int i = 0; i < (EDGE/THREADS)*(EDGE/THREADS); i++){
        ret += sum[i];
    }
    return -(ret>>1);
}

int relation (int a, int b) {
    switch (b-a) {
        case 0:       return 8;
        case -EDGE-1: return 0;
        case -EDGE:   return 1;
        case -EDGE+1: return 2;
        case -1:      return 3;
        case 1:       return 4;
        case EDGE-1:  return 5;
        case EDGE:    return 6;
        case EDGE+1:  return 7;
        default:      return -1;
    }
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
    int *couplings, *couplings_buf, *spins_buf, *spins, *sum_result, *sum_result_buf;
    couplings = (int*)malloc(N*9*sizeof(int));
    spins = (int*)malloc(REPLICA*N*sizeof(int));
    memset(couplings, '\0', N*9*sizeof(int));
    sum_result = (int*)malloc((EDGE/THREADS)*(EDGE/THREADS)*sizeof(int));
    CUDA_CALL( cudaMalloc(&couplings_buf, N*9*sizeof(int)) );
    CUDA_CALL( cudaMalloc(&spins_buf, REPLICA*N*sizeof(int)) );
    CUDA_CALL( cudaMalloc(&sum_result_buf, (EDGE/THREADS)*(EDGE/THREADS)*sizeof(int)) );

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

    // work group division
    dim3 grid(EDGE/THREADS, EDGE/THREADS), block(THREADS, THREADS);

    // temperature
    float increase = (3.0 - 0.1) / (float)REPLICA;
    float betas[REPLICA];
    float beta = 0.1;
    for (int i = 0; i < REPLICA; i++) {
        betas[i] = beta;
        beta += increase;
    }

    int results[TIMES] = {0};
    printf("Start Annealing\n");
    for (int x = 0; x < TIMES; x++) {
        prepare_spins<<<grid, block>>>(spins_buf, randvals);
        for (int s = 0; s < SWEEP; s++) {
            for (int r = 0; r < REPLICA; r++) {
                for (int signal = 0; signal < 4; signal++) {
                    ising<<<grid, block>>>(spins_buf, couplings_buf, 
                                           signal, betas[r], randvals, r);
                }
            }
            // Parellel Tempering
            for (int r = 0; r < REPLICA-1; r++) {
                int e1 = reduced_sum(spins_buf, couplings_buf, r, sum_result, sum_result_buf);
                int e2 = reduced_sum(spins_buf, couplings_buf, r+1, sum_result, sum_result_buf);
                float u = log(rand()/(float)RAND_MAX);
                if (u < (e2-e1)*(betas[r+1]-betas[r])) {
                    swap<<<grid, block>>>(spins_buf, r);
                }
            }
        }

        int minE = 0;
        for (int r = 0; r < REPLICA; r++) {
            int e = reduced_sum(spins_buf, couplings_buf, r, sum_result, sum_result_buf);
            if (e < minE)
                minE = e;
        }
        results[x] = minE;
    }

    printf("Finish Annealing\n");

    // Write results to file
    FILE *output;
    output = fopen("output.txt", "w");
    for (int i = 0; i < TIMES; i++)
        fprintf(output, "%d\n", results[i]);
    fclose(output);

    // Release Objects
    free(couplings);
    free(spins);
    cudaFree(spins_buf);
    cudaFree(couplings_buf);
    cudaFree(sum_result);
    return 0;
}
