# CS239
Parallel Computing

How to compile and run:
Using the terminal, cd to src then run:
nvcc -o kmeans main.cu utils/utils.cu utils/runner.cu kernels/kernels.cu utils/mnish_dataloader.cpp -I. -lcudart
* note: 
In my computer, this line in main.cu doesn't work:
MNISTDataLoader loader("./data/MNIST/raw/train-images-idx3-ubyte", "./data/MNIST/raw/train-labels-idx1-ubyte");
But, by changing the path to absolute, it does:
 MNISTDataLoader loader("D:\\Jim\\UP\\MEngg in AI\\CS 239\\Project\\Code\\Parallel-Computing-Kmeans\\cuda-kmeans\\data\\MNIST\\raw\\train-images-idx3-ubyte",
            "D:\\Jim\\UP\\MEngg in AI\\CS 239\\Project\\Code\\Parallel-Computing-Kmeans\\cuda-kmeans\\data\\MNIST\\raw\\train-labels-idx1-ubyte" );
        
Run compiled executable code:
.\kmeans.exe