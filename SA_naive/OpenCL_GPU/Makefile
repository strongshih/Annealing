CC = nvcc

all: Ising-opencl

%-opencl: %.c
    $(CC) $< -o $@ -g -lOpenCL

clean: 
    rm -rf Ising-opencl
