CC = nvcc

all: Bifurcation-cuda

%-cuda: %.cu
	$(CC) $< -o $@ -g 

test:
	./Bifurcation-cuda 0.txt

clean: 
	rm -rf Bifurcation-cuda
