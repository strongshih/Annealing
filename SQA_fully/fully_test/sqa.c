#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define N 2048
#define THREADS 4
#define TIMES 10

#define M 16  // trotter layers
#define STEP 100

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
    char *couplings;
    couplings = (char*)malloc(N*N*sizeof(char));
    memset(couplings, '\0', N*N*sizeof(char));
    
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

    // initialize spins
    char *spins;
    spins = (char*)malloc(M*N*sizeof(char));
    
    // initialize 
    int *sigma;
    sigma = (int*)malloc(M*N*sizeof(int));
    
    int results[TIMES] = {0};
    int delta;
    float increase = (8 - 1/(float)16) / (float)STEP;
    float G0 = 8.;
    
    for (int t = 0; t < TIMES; t++) {
        // initialize spins
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                spins[m*N + n] = (rand() / (float) RAND_MAX) > .5 ? 1 : -1;
            }
        }
        // initialize sigmas
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                sigma[m*N+n] = 0;
            }
            for (int n = 0; n < N; n++) {
                for (int i = 0; i < N; i++) {
                    sigma[m*N+n] += spins[m*N+i]*couplings[n*N+i];
                }
            }
        }
        
        double curr = 0.;
        float beta = 1/(float)16;
        
        for (int p = 0; p < STEP; p++) {
            float Gamma = G0*(1.-(float)p/(float)STEP);
            float J_perp = -0.5*log(tanh((Gamma/M)*beta))/beta;
            double begin = omp_get_wtime();
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    int idx = m*N+n;
                    int delta = sigma[idx];
                    int upper = (m == 0 ? M-1 : m-1);
                    int lower = (m == m-1 ? 0 : m+1);
                    delta = 2*M*spins[idx]*(delta - M*J_perp*(spins[upper*N+n] + spins[lower*N+n]));
                    if ( (-log(rand() / (float) RAND_MAX) / beta) > delta ) {
                        #pragma omp parallel for schedule(static, N/THREADS) 
                        for (int i = 0; i < N; i++)
                            sigma[m*N+i] = sigma[m*N+i] - 2*spins[idx]*couplings[n*N+i];
                        spins[idx] = -spins[idx];
                    }
                }
            }
            beta += increase;
            double end = omp_get_wtime();
            double duration = (double)(end-begin);
            curr += duration;
            
            int E = 0;
            for (int i = 0; i < N; i++)
                for (int j = i+1; j < N; j++)
                    E += -spins[i]*spins[j]*couplings[i*N+j];
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
    free(spins);
    free(sigma);
    
    return 0;
}