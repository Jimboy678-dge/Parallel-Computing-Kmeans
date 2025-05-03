#ifndef KERNELS_H
#define KERNELS_H

#include "cuda_runtime.h"
#include <iostream>
#include "device_launch_parameters.h"

using namespace std;

__global__ void kmeans_000000(
    uint8_t* images, //flatten images of size N,IMAGE_HEIGHT,IMAGE_WIDTH
    uint8_t N, // number of images = 6000?
    uint8_t IMAGE_HEIGHT, // image height = 28
    uint8_t IMAGE_WIDTH, // image width = 28
    uint8_t* N_cluster, // array to hold N cluster label
    uint8_t K, // k-means parameter
    int seed
); 

// Add other kernel definitions

#endif // KERNELS_H