CC=nvcc
LDFLAGS=-lcurand -g -arch=sm_75 -rdc=true

all: PT-cuda

%-cuda: %.cu
    $(CC) $< -o $@ $(LDFLAGS)

test:
    ./PT-cuda tests/1048576.txt

clean: 
    rm -rf PT-cuda
