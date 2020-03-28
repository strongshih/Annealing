CC=nvcc
LDFLAGS=-lcurand -g -arch=sm_75 -rdc=true

all: king-cuda

%-cuda: %.cu
	$(CC) $< -o $@ $(LDFLAGS)

test:
	./king-cuda tests/1048576.txt

clean: 
	rm -rf king-cuda
