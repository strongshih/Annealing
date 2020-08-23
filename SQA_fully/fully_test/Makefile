CC = nvcc

all: sqa-cuda sqa-omp

%-cuda: %.cu
	$(CC) $< -o $@ -g 

%-omp: %.c
	gcc $< -o $@ -g -lm -fopenmp

test:
	./sqa-cuda WK2000_1.rud
	OMP_NUM_THREADS=4 ./sqa-omp WK2000_1.rud

clean: 
	rm -rf sqa-cuda sqa-omp

### Nsight compute
# /usr/local/cuda-10.0/NsightCompute-1.0/nv-nsight-cu-cli -f -o output ./opt WK2000_1.rud
# /usr/local/cuda-10.0/NsightCompute-1.0/nv-nsight-cu output.nsight-cuprof-report
