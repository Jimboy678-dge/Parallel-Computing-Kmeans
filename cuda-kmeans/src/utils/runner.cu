#include "runner.cuh"

void BaseRunner::runKernel(
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
) {
    return;
}


// COMMON 
void BaseRunner::run(
    std::vector<std::vector<uint8_t>> images,
    uint8_t N, // number of images = 6000?
    uint8_t IMAGE_HEIGHT, // image height = 28
    uint8_t IMAGE_WIDTH, // image width = 28
    std::vector<uint8_t> labels //not really needed but why not?
) {
    // [TODO: bdmatabang]
    // [1] Implement code to flatten image
    //     Flatten  std::vector<std::vector<uint8_t>> images to uint8_t* images of size N,

    // [TODO: bdmatabang]
    // [2] Implement code to allocate cuda memory  for flatten images

    // [TODO: bdmatabang]
    // [3] Copy images data from host to device

    // [TODO: bdmatabang]
    // [4] Setup kernel grids and blocks

    // [TODO: bdmatabang]
    // [5] Allocate memory for result uint8_t* N_cluster, run kernel and copy result to N_cluster
    //     [Note from g.agluba ] use simple kernels for now to test
    // runKernel(dimGrid,dimBlock, images_d,  N_d, IMAGE_HEIGHT, IMAGE_WIDTH,N_cluster_d, K, seed)
    // cudaMemcpy(N_cluster, N_cluster_d, memSizeNCluster, cudaMemcpyDeviceToHost);

    // [TODO: bdmatabang]
    // [6] Free-up some memmory

    return;

};