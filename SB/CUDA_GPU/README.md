# CUDA SB

## TEST

- include nvcc library path

```
export PATH=/opt/cuda/bin${PATH:+:${PATH}}$
export LD_LIBRARY_PATH=/opt/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

- test

```
git clone https://github.com/strongshih/Annealing.git
cd Annealing/SB/CUDA_GPU/
make
make test
```

- file

```
m2_opt.cu  (float) --> original
m2_opt2.cu (float) --> tensor core
m2_opt3.cu (half)  --> tensor core
```

## Update reference

![](./stats/update.png)
