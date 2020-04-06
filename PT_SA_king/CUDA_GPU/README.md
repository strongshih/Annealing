## Parallel Tempering King graph SA CUDA on GPU

### Run and Test

```
make
make test
```

### nvprof

```
==20292== NVPROF is profiling process 20292, command: ./PT-cuda /home/sam/Annealing/SA_king/CUDA_GPU/tests/1048576.txt
Finish copying coefficients
Start Annealing
Finish Annealing
==20292== Profiling application: ./PT-cuda /home/sam/Annealing/SA_king/CUDA_GPU/tests/1048576.txt
==20292== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   75.84%  4.58386s     32000  143.25us  140.30us  197.36us  ising(int*, int*, int, float, unsigned int*, int)
                   21.41%  1.29406s     16000  80.878us  77.829us  98.887us  reduction(int*, int*, int, int*)
                    2.34%  141.23ms      5261  26.844us  26.402us  28.258us  swap(int*, int)
                    0.28%  16.647ms     16000  1.0400us     960ns  11.681us  [CUDA memcpy DtoH]
                    0.09%  5.1645ms         2  2.5823ms  446.37us  4.7182ms  [CUDA memcpy HtoD]
                    0.06%  3.3288ms        10  332.88us  329.62us  343.74us  prepare_spins(int*, unsigned int*)
      API calls:   93.61%  6.03970s     16002  377.43us  11.893us  30.851ms  cudaMemcpy
                    3.58%  230.85ms     53271  4.3330us  3.6870us  394.17us  cudaLaunchKernel
                    2.72%  175.43ms         4  43.857ms  72.784us  174.95ms  cudaMalloc
                    0.04%  2.6463ms     16000     165ns     124ns  383.38us  cudaPeekAtLastError
                    0.04%  2.4146ms         3  804.86us  1.6810us  2.1615ms  cudaFree
                    0.01%  388.63us         1  388.63us  388.63us  388.63us  cuDeviceTotalMem
                    0.00%  206.38us        96  2.1490us     200ns  86.617us  cuDeviceGetAttribute
                    0.00%  31.274us         1  31.274us  31.274us  31.274us  cuDeviceGetName
                    0.00%  4.9760us         1  4.9760us  4.9760us  4.9760us  cuDeviceGetPCIBusId
                    0.00%  1.9700us         3     656ns     198ns  1.4670us  cuDeviceGetCount
                    0.00%     966ns         2     483ns     205ns     761ns  cuDeviceGet
                    0.00%     338ns         1     338ns     338ns     338ns  cuDeviceGetUuid
```