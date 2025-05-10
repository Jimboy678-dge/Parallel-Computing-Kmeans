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

const int DEFAULT_REPLICATION = 3;


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
        size_t sharedMemorySize = (K * IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(float)) +
            (K * sizeof(int));


        // Allocate memory for global sums and counts
        size_t memSizeClusterSums = K * IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(float);
        size_t memSizeClusterCount = K * sizeof(int);

        int* K_d_count; float* K_cluster_d_sum;
        cudaMalloc((void**)&K_cluster_d_sum, memSizeClusterSums);
        cudaMalloc((void**)&K_d_count, memSizeClusterCount);

        cudaMemset(K_cluster_d_sum, 0.0f, memSizeClusterSums);
        cudaMemset(K_d_count, 0, memSizeClusterCount);

        // Launch the kmeans_100000 kernel
        kmeans_100000<<<dimGrid, dimBlock, sharedMemorySize >>>(
            images_d,
            N,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            K_cluster_d,
            K,
            centroids_d,
            K_cluster_d_sum,
            K_d_count,
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

        // Warm-Up Step
        std::cout << "Running warm-up..." << std::endl;
        K000000Runner warmupRunner000000 = K000000Runner();
        warmupRunner000000.run(images, images.size(), DEFAULT_IMAGE_HEIGTH, DEFAULT_IMAGE_WIDTH, labels, 1);
        cudaDeviceSynchronize(); // Ensure kernel execution is complete

        K100000Runner warmupRunner100000 = K100000Runner();
        warmupRunner100000.run(images, images.size(), DEFAULT_IMAGE_HEIGTH, DEFAULT_IMAGE_WIDTH, labels, 1);
        cudaDeviceSynchronize(); // Ensure kernel execution is complete
        std::cout << "Warm-up completed." << std::endl;

        // Timing variables
        std::vector<double> k000000_times;
        std::vector<double> k100000_times;

        

        // Run K000000Runner 5 times
        for (int i = 0; i < DEFAULT_REPLICATION; ++i) {
            K000000Runner runner = K000000Runner();
            auto start = std::chrono::high_resolution_clock::now();
            runner.run(images, images.size(), DEFAULT_IMAGE_HEIGTH, DEFAULT_IMAGE_WIDTH, labels);
            //cudaDeviceSynchronize(); // Ensure kernel execution is complete
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            k000000_times.push_back(elapsed.count());
            std::cout << "K000000Runner Run " << i + 1 << ": " << elapsed.count() << " seconds" << std::endl;
        }

        // Run K100000Runner 5 times
        for (int i = 0; i < DEFAULT_REPLICATION; ++i) {
            K100000Runner runner = K100000Runner();
            auto start = std::chrono::high_resolution_clock::now();
            runner.run(images, images.size(), DEFAULT_IMAGE_HEIGTH, DEFAULT_IMAGE_WIDTH, labels);
            //cudaDeviceSynchronize(); // Ensure kernel execution is complete
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            k100000_times.push_back(elapsed.count());
            std::cout << "K100000Runner Run " << i + 1 << ": " << elapsed.count() << " seconds" << std::endl;
        }

        // Calculate and display average execution times
        double k000000_avg = std::accumulate(k000000_times.begin(), k000000_times.end(), 0.0) / k000000_times.size();
        double k100000_avg = std::accumulate(k100000_times.begin(), k100000_times.end(), 0.0) / k100000_times.size();

        std::cout << "\nAverage Execution Time:" << std::endl;
        std::cout << "K000000Runner: " << k000000_avg << " seconds" << std::endl;
        std::cout << "K100000Runner: " << k100000_avg << " seconds" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
