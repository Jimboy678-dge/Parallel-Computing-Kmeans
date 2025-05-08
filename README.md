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
