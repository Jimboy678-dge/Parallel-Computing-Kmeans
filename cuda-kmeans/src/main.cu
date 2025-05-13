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
//      K400200 means optimized kernel with shared memory kmeans_400200, kmeans_X00000 X>1 for any other variants
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

class K001000Runner : public BaseRunner {
    public:
        void runKernel(
            dim3 dimGrid,
            dim3 dimBlock,
            uint8_t* images_d, //flatten images of size N X IMAGE_HEIGHT X IMAGE_WIDTH
            size_t N, // number of images = 6000?
            uint8_t IMAGE_HEIGHT, // image height = 28
            uint8_t IMAGE_WIDTH, // image width = 28
            uint8_t* K_cluster_d, // array to hold K cluster label
            uint8_t K, // k-means parameter
            float* centroids_d, // flatten centroids of size K X IMAGE_HEIGHT X IMAGE_WIDTH
            int max_iter
        ) {
            kmeans_001000 << <dimGrid, dimBlock >> > (
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


class K400200Runner : public BaseRunner {
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

        // Launch the kmeans_400200 kernel
        kmeans_400200<<<dimGrid, dimBlock, sharedMemorySize >>>(
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

class K401200Runner : public BaseRunner {
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

        // Launch the kmeans_401200 kernel
        kmeans_401200<<<dimGrid, dimBlock, sharedMemorySize >>>(
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

const int IMAGE_HEIGHT = 28;
const int IMAGE_WIDTH = 28;
const int IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
void kmeans_cpu(const std::vector<std::vector<float>>& images,
                std::vector<std::vector<float>>& centroids,
                std::vector<uint8_t>& K_cluster,
                int K,
                int max_iter) {

    int N = images.size();

    for (int iter = 0; iter < max_iter; ++iter) {
        // Step 1: Assign each sample to a cluster
        for (int idx = 0; idx < N; ++idx) {
            float minDistance = std::numeric_limits<float>::max();
            uint8_t bestCluster = 0;

            for (int k = 0; k < K; ++k) {
                float distance = 0.0f;
                for (int i = 0; i < IMAGE_SIZE; ++i) {
                    float diff = images[idx][i] - centroids[k][i];
                    distance += diff * diff;
                }
                if (distance < minDistance) {
                    minDistance = distance;
                    bestCluster = k;
                }
            }

            K_cluster[idx] = bestCluster;
        }

        // Step 2: Update centroids
        std::vector<std::vector<float>> newCentroids(K, std::vector<float>(IMAGE_SIZE, 0.0f));
        std::vector<int> clusterSizes(K, 0);

        for (int i = 0; i < N; ++i) {
            int cluster = K_cluster[i];
            clusterSizes[cluster]++;
            for (int j = 0; j < IMAGE_SIZE; ++j) {
                newCentroids[cluster][j] += images[i][j];
            }
        }

        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < IMAGE_SIZE; ++j) {
                if (clusterSizes[k] > 0) {
                    centroids[k][j] = newCentroids[k][j] / clusterSizes[k];
                }
                // Else keep the previous value (unchanged)
            }
        }
    }
}


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

        K400200Runner warmupRunner400200 = K400200Runner();
        warmupRunner400200.run(images, images.size(), DEFAULT_IMAGE_HEIGTH, DEFAULT_IMAGE_WIDTH, labels, 1);
        cudaDeviceSynchronize(); // Ensure kernel execution is complete
        std::cout << "Warm-up completed." << std::endl;

        // Timing variables
        std::vector<double> cpu_times;
        std::vector<double> k000000_times;
        std::vector<double> k001000_times;
        std::vector<double> k400200_times;
        std::vector<double> k401200_times;

        // Run CPUversion 5 times
        for (int i = 0; i < DEFAULT_REPLICATION; ++i) {
            std::vector<std::vector<float>> imagez(images.size(), std::vector<float>(IMAGE_HEIGHT * IMAGE_WIDTH));
            std::vector<std::vector<float>> centroidz(10, std::vector<float>(IMAGE_HEIGHT * IMAGE_WIDTH));
            std::vector<uint8_t> K_clusterz(images.size());
            auto start = std::chrono::high_resolution_clock::now();
            kmeans_cpu(imagez, centroidz, K_clusterz, 10, 100);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            cpu_times.push_back(elapsed.count());
            std::cout << "CPURunner Run " << i + 1 << ": " << elapsed.count() << " seconds" << std::endl;
        }


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

        // Run K001000Runner 5 times
        for (int i = 0; i < DEFAULT_REPLICATION; ++i) {
            K001000Runner runner = K001000Runner();
            auto start = std::chrono::high_resolution_clock::now();
            runner.run(images, images.size(), DEFAULT_IMAGE_HEIGTH, DEFAULT_IMAGE_WIDTH, labels);
            //cudaDeviceSynchronize(); // Ensure kernel execution is complete
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            k001000_times.push_back(elapsed.count());
            std::cout << "K001000Runner Run " << i + 1 << ": " << elapsed.count() << " seconds" << std::endl;
        }

        // Run K400200Runner 5 times
        for (int i = 0; i < DEFAULT_REPLICATION; ++i) {
            K400200Runner runner = K400200Runner();
            auto start = std::chrono::high_resolution_clock::now();
            runner.run(images, images.size(), DEFAULT_IMAGE_HEIGTH, DEFAULT_IMAGE_WIDTH, labels);
            //cudaDeviceSynchronize(); // Ensure kernel execution is complete
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            k400200_times.push_back(elapsed.count());
            std::cout << "K400200Runner Run " << i + 1 << ": " << elapsed.count() << " seconds" << std::endl;
        }

        // Run K401200Runner 5 times
        for (int i = 0; i < DEFAULT_REPLICATION; ++i) {
            K401200Runner runner = K401200Runner();
            auto start = std::chrono::high_resolution_clock::now();
            runner.run(images, images.size(), DEFAULT_IMAGE_HEIGTH, DEFAULT_IMAGE_WIDTH, labels);
            //cudaDeviceSynchronize(); // Ensure kernel execution is complete
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            k401200_times.push_back(elapsed.count());
            std::cout << "K401200Runner Run " << i + 1 << ": " << elapsed.count() << " seconds" << std::endl;
        }

        // Calculate and display average execution times
        double cpu_avg = std::accumulate(cpu_times.begin(), cpu_times.end(), 0.0) / cpu_times.size();
        double k000000_avg = std::accumulate(k000000_times.begin(), k000000_times.end(), 0.0) / k000000_times.size();
        double k001000_avg = std::accumulate(k001000_times.begin(), k001000_times.end(), 0.0) / k001000_times.size();
        double k400200_avg = std::accumulate(k400200_times.begin(), k400200_times.end(), 0.0) / k400200_times.size();
        double k401200_avg = std::accumulate(k401200_times.begin(), k401200_times.end(), 0.0) / k401200_times.size();
        std::cout << "\nAverage Execution Time:" << std::endl;
        std::cout << "CPURunner: " << cpu_avg << " seconds" << std::endl;
        std::cout << "K000000Runner: " << k000000_avg << " seconds" << std::endl;
        std::cout << "K001000Runner: " << k001000_avg << " seconds" << std::endl;
        std::cout << "K400200Runner: " << k400200_avg << " seconds" << std::endl;
        std::cout << "K401200Runner: " << k401200_avg << " seconds" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
