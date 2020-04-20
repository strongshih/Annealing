CC = nvcc

all: Bifurcation_m1-cuda Bifurcation_m2-cuda opt

%-cuda: %.cu
	$(CC) $< -o $@ -g 

opt: m2_opt.cu
	nvcc -lcublas -lineinfo -arch=sm_75 -Xptxas -O3,-v -use_fast_math m2_opt.cu -o opt -g

test:
	./opt WK2000_1.rud

clean: 
	rm -rf Bifurcation_m1-cuda Bifurcation_m2-cuda opt

### Nsight compute
# /usr/local/cuda-10.0/NsightCompute-1.0/nv-nsight-cu-cli -f -o output ./opt WK2000_1.rud
# /usr/local/cuda-10.0/NsightCompute-1.0/nv-nsight-cu output.nsight-cuprof-report
