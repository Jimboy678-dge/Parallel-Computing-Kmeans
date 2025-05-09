#include "runner.cuh"
#include <random>

const uint8_t DEFAULT_K = 10;
const int DEFAULT_BLOCK_WIDTH = 32;
const int MAX_ITERATION = 100; // [b.matabang note] lowered down to 100 iterations

void BaseRunner::runKernel(
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
) {
    return;
}


// COMMON 
void BaseRunner::run(
    std::vector<std::vector<uint8_t>> images,
    size_t N, // number of images = 6000?
    uint8_t IMAGE_HEIGHT, // image height = 28
    uint8_t IMAGE_WIDTH, // image width = 28
    std::vector<uint8_t> labels //not really needed but why not?
) {
    // [1] Process inputs and initialize arbitrary centroids
    size_t memSizeImages = N * IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(uint8_t);
    uint8_t* flattenedImages = flattenImages(images, N, IMAGE_HEIGHT, IMAGE_WIDTH);
    visImage(flattenedImages, N, IMAGE_HEIGHT, IMAGE_WIDTH, 3456);

    size_t memSizeCentroids = DEFAULT_K * IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(float);
    float* centroids = initCentroids(12345, DEFAULT_K, IMAGE_HEIGHT, IMAGE_WIDTH);
    visCentroids(centroids, DEFAULT_K, IMAGE_HEIGHT, IMAGE_WIDTH, 5);

    // [2] Implement code to allocate cuda memory  for images and centroids
    uint8_t* images_d; float* centroids_d;
    cudaMalloc((void**)&images_d, memSizeImages);
    cudaMalloc((void**)&centroids_d, memSizeCentroids);

    // [3] Copy images/centroids data from host to device
    cudaMemcpy(images_d, flattenedImages, memSizeImages, cudaMemcpyHostToDevice);
    cudaMemcpy(centroids_d, centroids, memSizeCentroids, cudaMemcpyHostToDevice);

    // [4] Setup kernel grids and blocks
    int gridSizeX = (N + DEFAULT_BLOCK_WIDTH - 1) / DEFAULT_BLOCK_WIDTH; // number of blocks per dim.y in grid
    dim3 dimGrid(gridSizeX); // maximize number
    dim3 dimBlock(DEFAULT_BLOCK_WIDTH);

    // [TODO: bdmatabang]
    // Perform some initial warmup, accessing the device.
    //   Add any kernel that access the device memory

    // [5] Allocate memory for result uint8_t* K_cluster, run kernel and copy result to K_cluster
    //     [Note from g.agluba ] use simple kernels for now to test
    size_t memSizeKCluster = N * sizeof(uint8_t);
    uint8_t* K_cluster_d = new uint8_t[memSizeKCluster];
    cudaMalloc((void**)&K_cluster_d, memSizeKCluster); // [b.matabang note] added allocation here!
    runKernel(dimGrid, dimBlock, images_d, N, IMAGE_HEIGHT, IMAGE_WIDTH, K_cluster_d, DEFAULT_K, centroids_d, MAX_ITERATION);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // [6] Copy results to host
    uint8_t* K_cluster = new uint8_t[memSizeKCluster];
    cudaMemcpy(K_cluster, K_cluster_d, memSizeKCluster, cudaMemcpyDeviceToHost);

    // [7] Free-up some memmory
    cudaFree(images_d); cudaFree(K_cluster_d); cudaFree(centroids_d);
    free(flattenedImages); free(centroids);free(K_cluster);
    
};


// Initializing K*IMAGE_HEIGHT*IMAGE_WIDTH centroids
float* BaseRunner::initCentroids(
    int seed,
    uint8_t K,
    uint8_t IMAGE_HEIGHT,
    uint8_t IMAGE_WIDTH
) {
    size_t totalSize = K * IMAGE_HEIGHT * IMAGE_WIDTH;
    float* centroids = new float[totalSize];

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 255.0f);

    for (size_t i = 0; i < totalSize;++i) {
        centroids[i] = dist(rng);
    }
    return centroids;
}

void BaseRunner::visCentroids(
    float* centroids,
    uint8_t K,
    uint8_t IMAGE_HEIGHT,
    uint8_t IMAGE_WIDTH,
    int index
) {
    for (int j = 0; j < IMAGE_HEIGHT; ++j) {
        for (int k = 0; k < IMAGE_WIDTH; ++k) {
            // Print pixel intensity (0-255) as a character
            int pixelValue = static_cast<int>(centroids[(index * IMAGE_HEIGHT * IMAGE_WIDTH) + (j * IMAGE_WIDTH) + k]);
            std::cout << (pixelValue > 127 ? "#" : "."); // Threshold for visualization
        }
        std::cout << std::endl;
    }
}


uint8_t* BaseRunner::flattenImages(
    std::vector<std::vector<uint8_t>> images,
    size_t N,
    uint8_t IMAGE_HEIGHT,
    uint8_t IMAGE_WIDTH
) {
    size_t dataSetTotalSize = N * IMAGE_HEIGHT * IMAGE_WIDTH;
    uint8_t* flattenImages = new uint8_t[dataSetTotalSize];

    for (size_t i = 0; i < N; i++) {
        const auto& image = images[i];
        for (size_t j = 0; j < IMAGE_HEIGHT; j++) {
            for (size_t k = 0; k < IMAGE_WIDTH; k++) {
                flattenImages[(i * IMAGE_HEIGHT * IMAGE_WIDTH) + (j * IMAGE_WIDTH) + k] = image[j * IMAGE_WIDTH + k];
            }
        }
    }

    return flattenImages;
}

void BaseRunner::visImage(
    uint8_t* images,
    size_t N,
    uint8_t IMAGE_HEIGHT,
    uint8_t IMAGE_WIDTH,
    size_t index
) {
    for (int j = 0; j < IMAGE_HEIGHT; ++j) {
        for (int k = 0; k < IMAGE_WIDTH; ++k) {
            // Print pixel intensity (0-255) as a character
            int pixelValue = static_cast<int>(images[(index * IMAGE_HEIGHT * IMAGE_WIDTH) + (j * IMAGE_WIDTH) + k]);
            std::cout << (pixelValue > 0 ? "#" : "."); // Threshold for visualization
        }
        std::cout << std::endl;
    }
}