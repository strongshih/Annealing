#include <iostream>
#include <vector>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "src/cuda_nmfa.h"

#define SAMPLES 1 // output spin sample some
#define N 1024
#define SWEEP 100
#define ALPHA 0.15
#define SIGMA 0.15
#define SEED 1234

using namespace std;
 
int main (int argc, char *argv[]) {
    // initialize parameters
    float *couplings;
    int *samples;
    couplings = (float*)malloc(N*N*sizeof(float));
    samples = (int*)malloc(N*SAMPLES*sizeof(int));
    memset(couplings, '\0', N*N*sizeof(int));

    // Read couplings file 
    FILE *instance = fopen(argv[1], "r");
    assert(instance != NULL);
    int a, b, w;
    fscanf(instance, "%d", &a);
    while (!feof(instance)) {
        fscanf(instance, "%d%d%d", &a, &b, &w);
        couplings[a * N + b] = (float)w;
        couplings[b * N + a] = (float)w;
    }
    fclose(instance);

    float increase = (8 - 1/(float)16) / (float)SWEEP;
 	float beta = 1/(float)16;
	vector<float> betas;
	for (int i = 0; i < SWEEP; ++i) {
		betas.push_back(beta);
		beta += increase;
	}

	float elapsed_time = gpu_nmfa(N, couplings, betas, 
			SAMPLES, samples, SIGMA, ALPHA, SEED);

	cout << elapsed_time << endl;

    // Release Objects
    free(samples);
    free(couplings);
    return 0;
}
