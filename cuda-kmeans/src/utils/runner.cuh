#ifndef RUNNER_H
#define RUNNER_H

#include <iostream>
//#include "utils.cuh"
//#include "kernel.cuh"
#include "cuda_runtime.h"
#include "vector"

// Base Runner Class to run kernels, to avoid code redundancy
class BaseRunner {
public:
    // ABSTRACT method -- implement kernel as your own
    virtual void runKernel(
        dim3 dimGrid,
        dim3 dimBlock,
        // [g.agluba note] 
        // typically use float, but since most data are positive int, will make unsigned int to reduce memory
        uint8_t* images, //flatten images of size N,IMAGE_HEIGHT,IMAGE_WIDTH
        uint8_t N, // number of images = 6000?
        uint8_t IMAGE_HEIGHT, // image height = 28
        uint8_t IMAGE_WIDTH, // image width = 28
        uint8_t* N_cluster, // array to hold N cluster label
        uint8_t K, // k-means parameter
        int seed
    );

    // COMMON 
    void run(
        std::vector<std::vector<uint8_t>> images,
        uint8_t N, // number of images = 6000?
        uint8_t IMAGE_HEIGHT, // image height = 28
        uint8_t IMAGE_WIDTH, // image width = 28
        std::vector<uint8_t> labels //not really needed but why not?
    );

};

#endif // RUNNER_H