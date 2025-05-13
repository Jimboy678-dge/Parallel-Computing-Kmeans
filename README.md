# CS239
Parallel Computing

## Compile and Run
1. Using the terminal, cd to src then run:
```
nvcc -o kmeans main.cu utils/utils.cu utils/runner.cu kernels/kernels.cu utils/mnish_dataloader.cpp -I. -lcudart
```
Note:
In main.cu, it is recommended to review the actual path called by MNISTDataLoader to avoid issues.
<br>     

2. Run compiled executable code:

```
.\kmeans.exe
```

## INITIAL RESULTS FROM BDM (100 iterations, Ave. of 5 Runs):
K000000 : 168.601 seconds
K001000 : 102.212 seconds
K101000 : 169.480 seconds (needs review)