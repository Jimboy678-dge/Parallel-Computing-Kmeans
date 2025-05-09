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


// LOOP UNROLLING OPTIMIZATION 001000
// [b.matabang notes]
// This kernel is a more optimized version using LOOP UNROLLING ONLY
// Known Issues:
// with Large N (6000), will likely yield incorrect or unstable results,
//       due to race conditions/lagging threads and lack of global sync.
// Possible Fixes:
// Use multiple kernels, one for assigning clusters and another for updating centroids

__global__ void kmeans_001000(
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
    for (int iter = 0; iter < max_iter; iter++) {
        if (idx < N) {
            float minDistance = FLT_MAX;
            uint8_t bestCluster = 0;

            // Assign sample to a cluster [000000] LINE 25
            for (int k = 0; k < K; k++) {
                if (k >= K) continue; // Checks for out of bounds! Safeguard hehe
                float distance = 0.0f;
                int i = 0;

                // Unroll loop from [000000] LINE 27 (IMAGE_HEIGHT * IMAGE_WIDTH)
                for (; i + 3 < IMAGE_HEIGHT * IMAGE_WIDTH; i += 4) {
                    float diff0 = images_d[idx * IMAGE_HEIGHT * IMAGE_WIDTH + i] - centroids_d[k * IMAGE_HEIGHT * IMAGE_WIDTH + i];
                    float diff1 = images_d[idx * IMAGE_HEIGHT * IMAGE_WIDTH + i + 1] - centroids_d[k * IMAGE_HEIGHT * IMAGE_WIDTH + i + 1];
                    float diff2 = images_d[idx * IMAGE_HEIGHT * IMAGE_WIDTH + i + 2] - centroids_d[k * IMAGE_HEIGHT * IMAGE_WIDTH + i + 2];
                    float diff3 = images_d[idx * IMAGE_HEIGHT * IMAGE_WIDTH + i + 3] - centroids_d[k * IMAGE_HEIGHT * IMAGE_WIDTH + i + 3];
                    distance += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
                }

                // Handle remaining elements
                for (; i < IMAGE_HEIGHT * IMAGE_WIDTH; i++) {
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
        if (idx < K) {
            float newCentroid[784] = { 0 }; // MNIST IMAGE_HEIGHT * IMAGE_WIDTH = 784
            int clusterSize = 0;             

            for (int i = 0; i < N; ++i) {
                if (K_cluster_d[i] >= K) continue; // Checks for out of bounds! Safeguard hehe
                if (K_cluster_d[i] == idx) {
                    int j = 0;

                    // Unroll loop from [000000] LINE 51 (IMAGE_HEIGHT * IMAGE_WIDTH)
                    for (; j + 7 < IMAGE_HEIGHT * IMAGE_WIDTH; j += 8) {
                        newCentroid[j] += images_d[i * IMAGE_HEIGHT * IMAGE_WIDTH + j];
                        newCentroid[j + 1] += images_d[i * IMAGE_HEIGHT * IMAGE_WIDTH + j + 1];
                        newCentroid[j + 2] += images_d[i * IMAGE_HEIGHT * IMAGE_WIDTH + j + 2];
                        newCentroid[j + 3] += images_d[i * IMAGE_HEIGHT * IMAGE_WIDTH + j + 3];
                        newCentroid[j + 4] += images_d[i * IMAGE_HEIGHT * IMAGE_WIDTH + j + 4];
                        newCentroid[j + 5] += images_d[i * IMAGE_HEIGHT * IMAGE_WIDTH + j + 5];
                        newCentroid[j + 6] += images_d[i * IMAGE_HEIGHT * IMAGE_WIDTH + j + 6];
                        newCentroid[j + 7] += images_d[i * IMAGE_HEIGHT * IMAGE_WIDTH + j + 7];
                        newCentroid[j + 8] += images_d[i * IMAGE_HEIGHT * IMAGE_WIDTH + j + 8];
                    }

                    // Handle remaining elements
                    for (; j < IMAGE_HEIGHT * IMAGE_WIDTH; j++) {
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

__global__ void kmeans_101000(
    uint8_t* images_d,
    size_t N,
    uint8_t IMAGE_HEIGHT,
    uint8_t IMAGE_WIDTH,
    uint8_t* K_cluster_d,
    uint8_t K,
    float* centroids_d,
    int max_iter
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int image_size = IMAGE_HEIGHT * IMAGE_WIDTH;

    extern __shared__ float shared_centroids[]; // Dynamic shared memory

    for (int iter = 0; iter < max_iter; iter++) {
        // Load centroids to shared memory
        int shared_idx = threadIdx.x;
        while (shared_idx < K * image_size) {
            shared_centroids[shared_idx] = centroids_d[shared_idx];
            shared_idx += blockDim.x;
        }
        __syncthreads(); // Ensure all threads loaded the centroids

        if (idx < N) {
            float minDistance = FLT_MAX;
            uint8_t bestCluster = 0;

            for (int k = 0; k < K; k++) {
                float distance = 0.0f;
                int i = 0;
                int base_img = idx * image_size;
                int base_cen = k * image_size;

                // Unrolled loop
                for (; i + 3 < image_size; i += 4) {
                    float diff0 = images_d[base_img + i] - shared_centroids[base_cen + i];
                    float diff1 = images_d[base_img + i + 1] - shared_centroids[base_cen + i + 1];
                    float diff2 = images_d[base_img + i + 2] - shared_centroids[base_cen + i + 2];
                    float diff3 = images_d[base_img + i + 3] - shared_centroids[base_cen + i + 3];
                    distance += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
                }
                for (; i < image_size; i++) {
                    float diff = images_d[base_img + i] - shared_centroids[base_cen + i];
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

        // Update step remains unchanged here (global memory)
        // You could use shared memory here too, but be careful about memory limits and atomicity
        if (idx < K) {
            float newCentroid[784] = {0};
            int clusterSize = 0;

            for (int i = 0; i < N; ++i) {
                if (K_cluster_d[i] == idx) {
                    for (int j = 0; j < image_size; ++j) {
                        newCentroid[j] += images_d[i * image_size + j];
                    }
                    clusterSize++;
                }
            }

            for (int j = 0; j < image_size; ++j) {
                centroids_d[idx * image_size + j] = clusterSize > 0 ? newCentroid[j] / clusterSize : centroids_d[idx * image_size + j];
            }
        }

        __syncthreads();
    }
}


