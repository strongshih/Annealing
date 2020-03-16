# OpenCL and CUDA on FPGA and GPU

## Environment

- CSIE Workstation meow1 with 3 RTX 2080 Ti
- Cyclone V FPGA

## Naive

### OpenCL on GPU

#### Execution

```
git clone https://github.com/strongshih/Annealing.git
cd Annealing/SA_naive/OpenCL_GPU/
make
./Ising-opencl kernel.cl ../../problems/1024.txt
```

#### Test and Plot

```
cd Annealing/SA_naive/OpenCL_GPU/tests/
./test.sh
```

### CUDA on GPU

#### Execution

```
git clone https://github.com/strongshih/Annealing.git
cd Annealing/SA_naive/CUDA_GPU/
make
./Ising-cuda ../../problems/512.txt
```

#### Test and Plot

```
cd Annealing/SA_naive/CUDA_GPU/tests/
./test.sh
```

### OpenCL on FPGA

```
git clone https://github.com/strongshih/Annealing.git
cd Annealing/SA_naive/OpenCL_FPGA/
make
make test
```

