CC = nvcc

all: Ising-cuda

%-cuda: %.cu
    $(CC) $< -o $@ -g 

clean: 
    rm -rf Ising-cuda
