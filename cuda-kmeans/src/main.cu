#include <iostream>
#include <filesystem>
#include <chrono> // For timing
#include <vector> // For storing execution times
#include <numeric> // For calculating averages
#include "utils/mnist_dataloader.h"
#include "utils/runner.cuh"
#include "utils/utils.cuh"
#include "kernels/kernels.cuh"
#include "cuda_runtime.h"

const uint8_t DEFAULT_IMAGE_WIDTH = 28; // MNIST images are 28x28
const uint8_t DEFAULT_IMAGE_HEIGTH = 28;


// [Note from g.agluba to all],
// Please check Runner / Kernel Naming convention
// Reference first:
//      Optimizations:
//          class 0: shared memory and tiling ?
//          class 1: warping
//          class 3. loop unrolling
//          class 4. parallel scan (according to some research, this is feasible  
//      ....  
//  THINK OF ANYTHING
// 
// Assuming we have 6 optimization (we can always change this),
//      K000000 means unoptimized code running based kernel kmeans_000000
//      K100000 means optimized kernel with shared memory kmeans_100000, kmeans_X00000 X>1 for any other variants
//      K101000 mean optimized kernel with shared memory and loop unrolling
//      ...


class K000000Runner : public BaseRunner {
public:
    void runKernel(
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
        kmeans_000000 << <dimGrid, dimBlock >> > (
            images_d,
            N,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            K_cluster_d,
            K,
            centroids_d,
            max_iter
        );
    }
};

// ADD Extended class here for other kernels
class K100000Runner : public BaseRunner {
public:
    void runKernel(
        dim3 dimGrid,
        dim3 dimBlock,
        uint8_t* images_d,
        size_t N,
        uint8_t IMAGE_HEIGHT,
        uint8_t IMAGE_WIDTH,
        uint8_t* K_cluster_d,
        uint8_t K,
        float* centroids_d,
        int max_iter
    ) {
        // Calculate shared memory size for centroids
        size_t sharedMemorySize = K * IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(float);

        // Launch the kmeans_100000 kernel
        kmeans_100000<<<dimGrid, dimBlock, sharedMemorySize>>>(
            images_d,
            N,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            K_cluster_d,
            K,
            centroids_d,
            max_iter
        );
    }
};


int main() {
    try {
        // Device Properties
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        std::cout << "Device Count\t" << deviceCount << std::endl;

        for (int device = 0; device < deviceCount; device++) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, device);
            printCudaDeviceProperties(deviceProp);
        }

        // Load Data
        MNISTDataLoader loader("../data/MNIST/raw/train-images-idx3-ubyte", "../data/MNIST/raw/train-labels-idx1-ubyte");
        loader.load();
        const auto& images = loader.getImages();
        const auto& labels = loader.getLabels();
        std::cout << "Loaded " << images.size() << " images and " << labels.size() << " labels." << std::endl;
        loader.visImg(3456); // visualize one image given data index, comment if you want

        // [todo g.agluba]
        // get command-line arguments for easier testing ... 
        // for now, edit this when testing

        // // Run kmeans_000000
        // K000000Runner runner000000 = K000000Runner();
        // runner000000.run(images, images.size(), DEFAULT_IMAGE_HEIGTH, DEFAULT_IMAGE_WIDTH, labels);

        // // Run kmeans_100000
        // K100000Runner runner100000 = K100000Runner();
        // runner100000.run(images, images.size(), DEFAULT_IMAGE_HEIGTH, DEFAULT_IMAGE_WIDTH, labels);

        // Warm-Up Step
        std::cout << "Running warm-up..." << std::endl;
        K100000Runner warmupRunner = K100000Runner();
        warmupRunner.run(images, images.size(), DEFAULT_IMAGE_HEIGTH, DEFAULT_IMAGE_WIDTH, labels);
        cudaDeviceSynchronize(); // Ensure kernel execution is complete
        std::cout << "Warm-up completed." << std::endl;

        // Timing variables
        std::vector<double> execution_times;

        // Run K100000Runner 5 times
        for (int i = 0; i < 5; ++i) {
            K100000Runner runner = K100000Runner();
            auto start = std::chrono::high_resolution_clock::now();
            runner.run(images, images.size(), DEFAULT_IMAGE_HEIGTH, DEFAULT_IMAGE_WIDTH, labels);
            cudaDeviceSynchronize(); // Ensure kernel execution is complete
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            execution_times.push_back(elapsed.count());
            std::cout << "K100000Runner Run " << i + 1 << ": " << elapsed.count() << " seconds" << std::endl;
        }

        // Calculate and display average execution time
        double average_time = std::accumulate(execution_times.begin(), execution_times.end(), 0.0) / execution_times.size();
        std::cout << "\nAverage Execution Time for K100000Runner: " << average_time << " seconds" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
