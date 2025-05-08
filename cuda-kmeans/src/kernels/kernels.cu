#include "kernels.cuh"

__global__ void kmeans_000000(
    uint8_t* images_d, //flatten images of size N X IMAGE_HEIGHT X IMAGE_WIDTH
    size_t N, // number of images = 6000?
    uint8_t IMAGE_HEIGHT, // image height = 28
    uint8_t IMAGE_WIDTH, // image width = 28
    uint8_t* K_cluster_d, // array to hold K cluster label
    uint8_t K, // k-means parameter
    float* centroids_d, // flatten centroids of size K X IMAGE_HEIGHT X IMAGE_WIDTH
    int max_iter
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // THIS IS SUPER NAIVE... 
    //    Thinking if I should make more optimized version as baseline.

    for (int iter = 0; iter < max_iter; iter++) {
        if (idx < N) {

            float minDistance = FLT_MAX;
            uint8_t bestCluster = 0;
            // Asssign sample to a cluster
            // [Optimization Notes.1] Centroids can be loaded via shared memory to increase data reuse
            for (int k = 0;k < K;k++) {
                float distance = 0.0f;
                for (int i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH; ++i) {
                    float diff = images_d[idx * IMAGE_HEIGHT * IMAGE_WIDTH + i] - centroids_d[k * IMAGE_HEIGHT * IMAGE_WIDTH + i];
                    distance += diff * diff;
                }
                if (distance < minDistance) {
                    minDistance = distance;
                    bestCluster = k;
                }
            }

            K_cluster_d[idx] = bestCluster;
        }
        __syncthreads();
        
        // Update centroids
        // [Optimization Notes 2.] Can be optimized by parallel scan
        if (idx < K) {
            float newCentroid[784] = { 0 }; // Assuming IMAGE_HEIGHT * IMAGE_WIDTH = 784
            int clusterSize = 0;

            for (int i = 0; i < N; ++i) {
                // NOT OPTIMIZE, only one iteration is executed
                // We can divide N to K threads
                if (K_cluster_d[i] == idx) {
                    for (int j = 0; j < IMAGE_HEIGHT * IMAGE_WIDTH; ++j) {
                        newCentroid[j] += images_d[i * IMAGE_HEIGHT * IMAGE_WIDTH + j];
                    }
                    clusterSize++;
                }
            }

            // Average the sum to get the new centroid
            for (int j = 0; j < IMAGE_HEIGHT * IMAGE_WIDTH; ++j) {
                centroids_d[idx * IMAGE_HEIGHT * IMAGE_WIDTH + j] = clusterSize > 0 ? newCentroid[j] / clusterSize : centroids_d[idx * IMAGE_HEIGHT * IMAGE_WIDTH + j];
            }
        }
        __syncthreads();

    }
}

__global__ void kmeans_100000(
    uint8_t* images_d, // Flattened images of size N X IMAGE_HEIGHT X IMAGE_WIDTH
    size_t N,          // Number of images
    uint8_t IMAGE_HEIGHT, // Image height
    uint8_t IMAGE_WIDTH,  // Image width
    uint8_t* K_cluster_d, // Array to hold K cluster labels
    uint8_t K,            // Number of clusters
    float* centroids_d,   // Flattened centroids of size K X IMAGE_HEIGHT X IMAGE_WIDTH
    int max_iter          // Maximum number of iterations
) {
    // Thread ID
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return; // Ensure thread does not exceed image count

    // Image dimensions
    size_t imageSize = IMAGE_HEIGHT * IMAGE_WIDTH;

    // Shared memory for centroids
    extern __shared__ float sharedCentroids[];

    for (int iter = 0; iter < max_iter; ++iter) {
        // Load centroids into shared memory
        for (int i = threadIdx.x; i < K * imageSize; i += blockDim.x) {
            sharedCentroids[i] = centroids_d[i];
        }
        __syncthreads();

        // Assignment Step: Find the nearest centroid
        float minDistance = FLT_MAX;
        uint8_t bestCluster = 0;

        for (uint8_t cluster = 0; cluster < K; ++cluster) {
            float distance = 0.0f;
            for (size_t pixel = 0; pixel < imageSize; ++pixel) {
                float diff = images_d[idx * imageSize + pixel] - sharedCentroids[cluster * imageSize + pixel];
                distance += diff * diff;
            }
            if (distance < minDistance) {
                minDistance = distance;
                bestCluster = cluster;
            }
        }

        // Assign the image to the best cluster
        K_cluster_d[idx] = bestCluster;
        __syncthreads();

        // Update Step: Compute new centroids
        if (threadIdx.x < K) {
            float newCentroid[784] = { 0 }; // Assuming IMAGE_HEIGHT * IMAGE_WIDTH = 784
            int clusterSize = 0;

            for (int i = 0; i < N; ++i) {
                if (K_cluster_d[i] == threadIdx.x) {
                    for (int j = 0; j < imageSize; ++j) {
                        newCentroid[j] += images_d[i * imageSize + j];
                    }
                    clusterSize++;
                }
            }

            // Average the sum to get the new centroid
            for (int j = 0; j < imageSize; ++j) {
                centroids_d[threadIdx.x * imageSize + j] = clusterSize > 0 ? newCentroid[j] / clusterSize : centroids_d[threadIdx.x * imageSize + j];
            }
        }
        __syncthreads();
    }
}