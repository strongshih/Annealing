# CUDA SB

## File

```
python fully.py --> generate fully-connected problems, e.g: 0.txt
```

## TEST

```
git clone https://github.com/strongshih/Annealing.git
cd Annealing/SB/CUDA_GPU/
make
make test
```

## Update reference

![](./update.png)

## TODO

```
git clone https://github.com/strongshih/Annealing.git
cd Annealing/SB/CUDA_GPU/
* modify Bifurcation.cu
```

- At least energy results will be consistent with the results given by https://arxiv.org/abs/1401.1084
- Use `nvvp` or `nvprof` to further analyze the code, and optimize it
- Hopefully, the codes can achieve same experiment results as Toshiba SB's