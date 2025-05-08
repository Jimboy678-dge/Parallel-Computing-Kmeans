# CS239
Parallel Computing

## Compile and Run
1. Using the terminal, cd to src then run:
```
nvcc -o kmeans main.cu utils/utils.cu utils/runner.cu kernels/kernels.cu utils/mnish_dataloader.cpp -I. -lcudart
```
<br>
Note: 
It is recommended to use absolute path when using MNISTDataLoader to avoid issues.  ex: <br>
```
MNISTDataLoader loader("D:\\Jim\\UP\\MEngg in AI\\CS 239\\Project\\Code\\Parallel-Computing-Kmeans\\cuda-kmeans\\data\\MNIST\\raw\\train-images-idx3-ubyte",
            "D:\\Jim\\UP\\MEngg in AI\\CS 239\\Project\\Code\\Parallel-Computing-Kmeans\\cuda-kmeans\\data\\MNIST\\raw\\train-labels-idx1-ubyte" );
```
<br>     
2. Run compiled executable code:
.\kmeans.exe
