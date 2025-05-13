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

## INITIAL RESULTS FROM BDM (100 iterations, Ave. of 3 Runs):
CPURunner: 543.231 seconds <br>
K000000Runner: 174.02 seconds <br>
K001000Runner: 102.297 seconds <br>
K400200Runner: 5.02361 seconds <br>
K401200Runner: 5.06317 seconds