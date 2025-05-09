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
        uint8_t* images_d, //flatten images of size N X IMAGE_HEIGHT X IMAGE_WIDTH
        size_t N, // number of images = 6000?
        uint8_t IMAGE_HEIGHT, // image height = 28
        uint8_t IMAGE_WIDTH, // image width = 28
        uint8_t* K_cluster_d, // array to hold K cluster label
        uint8_t K, // k-means parameter
        float* centroids_d, // flatten centroids of size K X IMAGE_HEIGHT X IMAGE_WIDTH
        int max_iter
    );

    // COMMON
    
    void runwarmup(
        std::vector<std::vector<uint8_t>> images,
        size_t N, // number of images = 6000?
        uint8_t IMAGE_HEIGHT, // image height = 28
        uint8_t IMAGE_WIDTH, // image width = 28
        std::vector<uint8_t> labels //not really needed but why not?
    );

    void run(
        std::vector<std::vector<uint8_t>> images,
        size_t N, // number of images = 6000?
        uint8_t IMAGE_HEIGHT, // image height = 28
        uint8_t IMAGE_WIDTH, // image width = 28
        std::vector<uint8_t> labels //not really needed but why not?
    );

    float* initCentroids(
        int seed,
        uint8_t K,
        uint8_t IMAGE_HEIGHT,
        uint8_t IMAGE_WIDTH
    );

    void BaseRunner::visCentroids(
        float* centroids,
        uint8_t K,
        uint8_t IMAGE_HEIGHT,
        uint8_t IMAGE_WIDTH,
        int index
    );

    uint8_t* BaseRunner::flattenImages(
        std::vector<std::vector<uint8_t>> images,
        size_t N,
        uint8_t IMAGE_HEIGHT,
        uint8_t IMAGE_WIDTH
    );

    void BaseRunner::visImage(
        uint8_t* images,
        size_t N,
        uint8_t IMAGE_HEIGHT,
        uint8_t IMAGE_WIDTH,
        size_t index
    );

};

#endif // RUNNER_H