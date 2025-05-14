#include "kernels.cuh"
#include <device_atomic_functions.h>

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
    // printf("number of pixels: %d\n", IMAGE_HEIGHT * IMAGE_WIDTH); // Debugging line
    // printf("number of clusters: %d\n", K); // Debugging 
    // printf("K_cluster_d: %d\n", K_cluster_d); // Debugging line

    // THIS IS SUPER NAIVE... 
    //    Thinking if I should make more optimized version as baseline.
    // printf("Number of images: %d\n", N); // Debugging line
    for (int iter = 0; iter < max_iter; iter++) {
        // if (idx > 15000){
        if (idx < 0){
            // printf("Thread ID: %d, Iteration: %d\n", idx, iter); // Debugging line
            // printf("Number of images: %d\n", N);
        }
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
                    // if (k < 10 && i < 2 && iter < 2 && idx < 2) {
                    if (k < 10 && i < 2 && iter < 2 && idx < 10) {
                        // Debugging line to check the first distance calculation
                        // printf("Distance (global): %f\n, cluster: %d\n pixel: %d\n, iter: %d\n", distance, k, i, iter); // Debugging line
                    }
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
    uint8_t* images_d,
    size_t N,
    uint8_t IMAGE_HEIGHT,
    uint8_t IMAGE_WIDTH,
    uint8_t* K_cluster_d,
    uint8_t K,
    float* centroids_d,
    int max_iter
) {
    extern __shared__ float shared_centroids[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int image_size = IMAGE_HEIGHT * IMAGE_WIDTH;

    for (int iter = 0; iter < max_iter; iter++) {
        if (idx < 2){
            // printf("Thread ID: %d, Iteration: %d\n", tid, iter); // Debugging line
        }
        // printf("Thread ID: %d, Iteration: %d\n", tid, iter); // Debugging line
        // Load centroids into shared memory (by threads cooperatively)
        /*for (int i = tid; i < K * image_size; i += blockDim.x) {
            shared_centroids[i] = centroids_d[i];
        }*/
        // More optimize [distributed to all threads == griDim.x * blockDim.x]
        for (int i = idx; i < K * image_size; i += gridDim.x * blockDim.x) {
            shared_centroids[i] = centroids_d[i];
        }

        __syncthreads();

        // Step 1: Assign images to nearest cluster
        if (idx < N) {
            float min_dist = FLT_MAX;
            uint8_t best_cluster = 0;

            for (int k = 0; k < K; ++k) {
                float dist = 0.0f;
                for (int j = 0; j < image_size; ++j) {
                    float diff = static_cast<float>(images_d[idx * image_size + j]) - shared_centroids[k * image_size + j];
                    dist += diff * diff;
                    if (k < 2 && j < 2  && iter < 2 && idx < 2) {
                        // Debugging line to check the first distance calculation
                        // cout << "Distance (shared): " << distance << endl; // Debugging line
                        // printf("Distance (shared): %f\n, cluster: %d\n pixel: %d\n, iter: %d\n", dist, k, j, iter); // Debugging line
                    }
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }
            }

            K_cluster_d[idx] = best_cluster;
        }
        __syncthreads();

        // Step 2: Update centroids (one block handles all clusters)
        if (idx < K) {
            float new_centroid[784] = { 0.0f }; // Assumes max image size
            int count = 0;

            for (int i = 0; i < N; ++i) {
                if (K_cluster_d[i] == idx) {
                    for (int j = 0; j < image_size; ++j) {
                        new_centroid[j] += static_cast<float>(images_d[i * image_size + j]);
                    }
                    count++;
                }
            }

            for (int j = 0; j < image_size; ++j) {
                if (count > 0)
                    centroids_d[idx * image_size + j] = new_centroid[j] / count;
                // else keep previous value
            }
        }
        __syncthreads();
    }
}

// 1 image per block
// from ChatGPT
__global__ void kmeans_200000(
    uint8_t* images_d,
    size_t N,
    uint8_t IMAGE_HEIGHT,
    uint8_t IMAGE_WIDTH,
    uint8_t* K_cluster_d,
    uint8_t K,
    float* centroids_d,
    int max_iter
) {
    extern __shared__ uint8_t shared_image[]; // Shared memory for one image per block

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int image_size = IMAGE_HEIGHT * IMAGE_WIDTH;

    if (idx < N) {
            // Load the current thread's image to shared memory (1 image per block)
            for (int i = tid; i < image_size; i += blockDim.x) {
                shared_image[i] = images_d[idx * image_size + i];
            }
            __syncthreads();
    }

    for (int iter = 0; iter < max_iter; iter++) {
        if (idx < N) {

            float minDistance = FLT_MAX;
            uint8_t bestCluster = 0;

            for (int k = 0; k < K; k++) {
                float distance = 0.0f;
                for (int j = 0; j < image_size; ++j) {
                    float diff = static_cast<float>(shared_image[j]) - centroids_d[k * image_size + j];
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

        // Update step (naive version)
        if (idx < K) {
            float newCentroid[784] = {0};
            int clusterSize = 0;

            for (int i = 0; i < N; ++i) {
                if (K_cluster_d[i] == idx) {
                    for (int j = 0; j < image_size; ++j) {
                        newCentroid[j] += static_cast<float>(images_d[i * image_size + j]);
                    }
                    clusterSize++;
                }
            }

            for (int j = 0; j < image_size; ++j) {
                if (clusterSize > 0)
                    centroids_d[idx * image_size + j] = newCentroid[j] / clusterSize;
            }
        }
        __syncthreads();
    }
}

// 32 images per block
// from ChatGPT
__global__ void kmeans_300000(
    uint8_t* images_d,
    size_t N,
    uint8_t IMAGE_HEIGHT,
    uint8_t IMAGE_WIDTH,
    uint8_t* K_cluster_d,
    uint8_t K,
    float* centroids_d,
    int max_iter
) {
    const int image_size = IMAGE_HEIGHT * IMAGE_WIDTH;
    const int images_per_block = 32;
    const int tid = threadIdx.x;
    const int block_image_idx = blockIdx.x * images_per_block;
    const int global_image_idx = block_image_idx + tid;

    // Shared memory for 32 images
    extern __shared__ uint8_t shared_images[];

    // Load 32 images into shared memory
    if (tid < images_per_block && global_image_idx < N) {
        for (int i = 0; i < image_size; ++i) {
            shared_images[tid * image_size + i] =
                images_d[global_image_idx * image_size + i];
        }
    }
    __syncthreads();

    for (int iter = 0; iter < max_iter; iter++) {
        

        // Each thread assigns one image to the nearest cluster
        if (tid < images_per_block && global_image_idx < N) {
            float min_dist = FLT_MAX;
            uint8_t best_cluster = 0;

            for (int k = 0; k < K; ++k) {
                float dist = 0.0f;
                for (int j = 0; j < image_size; ++j) {
                    float diff = static_cast<float>(
                        shared_images[tid * image_size + j]) -
                        centroids_d[k * image_size + j];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }
            }
            K_cluster_d[global_image_idx] = best_cluster;
        }
        __syncthreads();

        // Update centroids (one thread per centroid)
        if (global_image_idx < K) {
            float new_centroid[784] = {0};
            int count = 0;
            for (int i = 0; i < N; ++i) {
                if (K_cluster_d[i] == global_image_idx) {
                    for (int j = 0; j < image_size; ++j) {
                        new_centroid[j] += static_cast<float>(
                            images_d[i * image_size + j]);
                    }
                    count++;
                }
            }

            for (int j = 0; j < image_size; ++j) {
                if (count > 0)
                    centroids_d[global_image_idx * image_size + j] =
                        new_centroid[j] / count;
            }
        }
        __syncthreads();
    }
}

// Loop Unrolling only
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
    extern __shared__ float shared_centroids[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int image_size = IMAGE_HEIGHT * IMAGE_WIDTH;

    for (int iter = 0; iter < max_iter; iter++) {
        // Load centroids into shared memory
        for (int i = idx; i < K * image_size; i += gridDim.x * blockDim.x) {
            shared_centroids[i] = centroids_d[i];
        }

        __syncthreads();

        // Step 1: Assign images to nearest cluster
        if (idx < N) {
            float min_dist = FLT_MAX;
            uint8_t best_cluster = 0;

            for (int k = 0; k < K; ++k) {
                float dist = 0.0f;
                int base_img = idx * image_size;
                int base_cent = k * image_size;

                // Manual unroll (factor = 4)
                for (int j = 0; j + 3 < image_size; j += 4) {
                    float d0 = (float)images_d[base_img + j]     - shared_centroids[base_cent + j];
                    float d1 = (float)images_d[base_img + j + 1] - shared_centroids[base_cent + j + 1];
                    float d2 = (float)images_d[base_img + j + 2] - shared_centroids[base_cent + j + 2];
                    float d3 = (float)images_d[base_img + j + 3] - shared_centroids[base_cent + j + 3];
                    dist += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                }

                // Handle remainder
                for (int j = (image_size / 4) * 4; j < image_size; ++j) {
                    float diff = (float)images_d[base_img + j] - shared_centroids[base_cent + j];
                    dist += diff * diff;
                }

                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }
            }

            K_cluster_d[idx] = best_cluster;
        }
        __syncthreads();

        // Step 2: Update centroids
        if (idx < K) {
            float new_centroid[784] = { 0.0f }; // assumes max image size
            int count = 0;

            for (int i = 0; i < N; ++i) {
                if (K_cluster_d[i] == idx) {
                    int base_img = i * image_size;

                    // Manual unroll (factor = 4)
                    for (int j = 0; j + 3 < image_size; j += 4) {
                        new_centroid[j]     += (float)images_d[base_img + j];
                        new_centroid[j + 1] += (float)images_d[base_img + j + 1];
                        new_centroid[j + 2] += (float)images_d[base_img + j + 2];
                        new_centroid[j + 3] += (float)images_d[base_img + j + 3];
                    }

                    // Remainder
                    for (int j = (image_size / 4) * 4; j < image_size; ++j) {
                        new_centroid[j] += (float)images_d[base_img + j];
                    }

                    count++;
                }
            }

            int base_cent = idx * image_size;
            // Normalize new centroid (manual unroll)
            for (int j = 0; j + 3 < image_size; j += 4) {
                if (count > 0) {
                    centroids_d[base_cent + j]     = new_centroid[j] / count;
                    centroids_d[base_cent + j + 1] = new_centroid[j + 1] / count;
                    centroids_d[base_cent + j + 2] = new_centroid[j + 2] / count;
                    centroids_d[base_cent + j + 3] = new_centroid[j + 3] / count;
                }
            }

            for (int j = (image_size / 4) * 4; j < image_size; ++j) {
                if (count > 0) {
                    centroids_d[base_cent + j] = new_centroid[j] / count;
                }
            }
        }
        __syncthreads();
    }
}

__global__ void kmeans_400200(
    uint8_t* images_d,
    size_t N,
    uint8_t IMAGE_HEIGHT,
    uint8_t IMAGE_WIDTH,
    uint8_t* K_cluster_d,
    uint8_t K,
    float* centroids_d,
    float* K_cluster_d_sum,
    int* K_d_count,
    int max_iter
) {
    extern __shared__ float shared_centroids[];
    extern __shared__ int shared_count[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int image_size = IMAGE_HEIGHT * IMAGE_WIDTH;

    for (int iter = 0; iter < max_iter; iter++) {
        if (idx < 2) {
            // printf("Thread ID: %d, Iteration: %d\n", tid, iter); // Debugging line
        }
        // printf("Thread ID: %d, Iteration: %d\n", tid, iter); // Debugging line
        // Load centroids into shared memory (by threads cooperatively)
        for (int i = tid; i < K * image_size; i += blockDim.x) {
            shared_centroids[i] = centroids_d[i];
        }

        __syncthreads();

        // Step 1: Assign images to nearest cluster
        float min_dist = FLT_MAX;
        uint8_t best_cluster = 0;
        if (idx < N) {
            for (int k = 0; k < K; ++k) {
                float dist = 0.0f;
                for (int j = 0; j < image_size; ++j) {
                    float diff = static_cast<float>(images_d[idx * image_size + j]) - shared_centroids[k * image_size + j];
                    dist += diff * diff;
                }
                /*if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }*/
                bool flag = (dist < min_dist);
                min_dist = flag ? dist : min_dist;
                best_cluster = flag ? k : best_cluster;
            }

            K_cluster_d[idx] = best_cluster;
        }
        __syncthreads();

        // Step 2: Update centroids (one block handles all clusters)
        //if (idx < K) {
        //    float new_centroid[784] = { 0.0f }; // Assumes max image size
        //    int count = 0;

        //    for (int i = 0; i < N; ++i) {
        //        if (K_cluster_d[i] == idx) {
        //            for (int j = 0; j < image_size; ++j) {
        //                new_centroid[j] += static_cast<float>(images_d[i * image_size + j]);
        //            }
        //            count++;
        //        }
        //    }

        //    for (int j = 0; j < image_size; ++j) {
        //        if (count > 0)
        //            centroids_d[idx * image_size + j] = new_centroid[j] / count;
        //        // else keep previous value
        //    }
        //}
        //__syncthreads();

        // REUSE shared_centroids for partial sums
        // Initialize partial sum and counts, collabaratively within the block
        for (int i = tid; i < K * image_size; i += blockDim.x) {
            shared_centroids[i] = 0.0f;
        }
        __syncthreads();
        if (tid < K) {
            shared_count[tid] = 0;
        }
        __syncthreads();

        // Accumulate local sums and counts in shared memory
        // 32 threads/block should be negligble for atomicAdd
        // Intra-Block Summation
        if (idx < N) {
            for (int i = 0;i < image_size;++i) {
                atomicAdd(&shared_count[best_cluster], 1);
                atomicAdd(&shared_centroids[best_cluster * image_size + i], images_d[idx * image_size + i]);
            }
        }
        __syncthreads();

        // Global Reduction
        // Accumulate all centroid sums/counts with global memory (collaboratively over block)
        for (int i = tid; i < K * image_size; i += blockDim.x) {
            atomicAdd(&K_cluster_d_sum[i], shared_centroids[i]);
        }
        __syncthreads();
        if (tid < K) {
            atomicAdd(&K_d_count[tid], shared_count[tid]);
        }
        __syncthreads();

        // Update centroids
        // Update new centroids with sum over counts (collaboratively over all threads)
        if (idx < K * image_size) {
            centroids_d[idx] = K_cluster_d_sum[idx] / K_d_count[idx/image_size];
        }

    }
};






// Shared Memory + Global Reduction + Loop Unrolling
__global__ void kmeans_401200(
    uint8_t* images_d,
    size_t N,
    uint8_t IMAGE_HEIGHT,
    uint8_t IMAGE_WIDTH,
    uint8_t* K_cluster_d,
    uint8_t K,
    float* centroids_d,
    float* K_cluster_d_sum,
    int* K_d_count,
    int max_iter
) {
    extern __shared__ float shared_centroids[];
    extern __shared__ int shared_count[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int image_size = IMAGE_HEIGHT * IMAGE_WIDTH;

    for (int iter = 0; iter < max_iter; iter++) {
        if (idx < 2) {
            // printf("Thread ID: %d, Iteration: %d\n", tid, iter); // Debugging line
        }
        // printf("Thread ID: %d, Iteration: %d\n", tid, iter); // Debugging line
        // Load centroids into shared memory (by threads cooperatively)
        
        
        for (int i = tid; i < K * image_size; i += blockDim.x) {
            shared_centroids[i] = centroids_d[i];
        }

        __syncthreads();

        // Step 1: Assign images to nearest cluster
        float min_dist = FLT_MAX;
        uint8_t best_cluster = 0;
        if (idx < N) {
            for (int k = 0; k < K; ++k) {
                float dist = 0.0f;
                int j = 0;
                
                // Unroll the loop for 4 iterations
                for (; j + 3 < 784; j += 4) {
                    float diff0 = static_cast<float>(images_d[idx * 784 + j + 0]) - shared_centroids[k * 784 + j + 0];
                    float diff1 = static_cast<float>(images_d[idx * 784 + j + 1]) - shared_centroids[k * 784 + j + 1];
                    float diff2 = static_cast<float>(images_d[idx * 784 + j + 2]) - shared_centroids[k * 784 + j + 2];
                    float diff3 = static_cast<float>(images_d[idx * 784 + j + 3]) - shared_centroids[k * 784 + j + 3];

                    dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
                }

                // Handle remaining elements
                for (; j < 784; j++) {
                    float diff = static_cast<float>(images_d[idx * 784 + j + 0]) - shared_centroids[k * 784 + j];
                    dist += diff * diff;
                }

                /*if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }*/
                bool flag = (dist < min_dist);
                min_dist = flag ? dist : min_dist;
                best_cluster = flag ? k : best_cluster;
            }

            K_cluster_d[idx] = best_cluster;
        }
        __syncthreads();

        // Step 2: Update centroids (one block handles all clusters)
        //if (idx < K) {
        //    float new_centroid[784] = { 0.0f }; // Assumes max image size
        //    int count = 0;

        //    for (int i = 0; i < N; ++i) {
        //        if (K_cluster_d[i] == idx) {
        //            for (int j = 0; j < image_size; ++j) {
        //                new_centroid[j] += static_cast<float>(images_d[i * image_size + j]);
        //            }
        //            count++;
        //        }
        //    }

        //    for (int j = 0; j < image_size; ++j) {
        //        if (count > 0)
        //            centroids_d[idx * image_size + j] = new_centroid[j] / count;
        //        // else keep previous value
        //    }
        //}
        //__syncthreads();

        // REUSE shared_centroids for partial sums
        // Initialize partial sum and counts, collabaratively within the block
        
        // partial unrolling
        int i = tid;
        for (; i + blockDim.x * 3 < K * image_size; i += blockDim.x * 4) {
            shared_centroids[i]                   = 0.0f;
            shared_centroids[i + blockDim.x]     = 0.0f;
            shared_centroids[i + blockDim.x * 2] = 0.0f;
            shared_centroids[i + blockDim.x * 3] = 0.0f;
        }

        // Handle remaining elements
        for (; i < K * image_size; i += blockDim.x) {
            shared_centroids[i] = 0.0f;
        }

        __syncthreads();
        if (tid < K) {
            shared_count[tid] = 0;
        }
        __syncthreads();

        // Accumulate local sums and counts in shared memory
        // 32 threads/block should be negligble for atomicAdd
        // Intra-Block Summation
        if (idx < N) {
            for (int i = 0;i < image_size;++i) {
                atomicAdd(&shared_count[best_cluster], 1);
                atomicAdd(&shared_centroids[best_cluster * image_size + i], images_d[idx * image_size + i]);
            }
        }
        __syncthreads();

        // Global Reduction
        // Accumulate all centroid sums/counts with global memory (collaboratively over block)
        for (int i = tid; i < K * image_size; i += blockDim.x) {
            atomicAdd(&K_cluster_d_sum[i], shared_centroids[i]);
        }
        __syncthreads();
        if (tid < K) {
            atomicAdd(&K_d_count[tid], shared_count[tid]);
        }
        __syncthreads();

        // Update centroids
        // Update new centroids with sum over counts (collaboratively over all threads)
        if (idx < K * image_size) {
            centroids_d[idx] = K_cluster_d_sum[idx] / K_d_count[idx/image_size];
        }

    }
}


__global__ void kmeans_400000(
    uint8_t* images_d,
    size_t N,
    uint8_t IMAGE_HEIGHT,
    uint8_t IMAGE_WIDTH,
    uint8_t* K_cluster_d,
    uint8_t K,
    float* centroids_d,
    int max_iter
) {
    // extern __shared__ float shared_centroids[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int image_size = IMAGE_HEIGHT * IMAGE_WIDTH;


    // Shared memory for 32 images
    extern __shared__ uint8_t shared_images[];
    if (tid < 32 && idx < N) {
        for (int i = 0; i < image_size; ++i) {
            shared_images[tid * image_size + i] =
                images_d[idx * image_size + i];
        }
    }
    __syncthreads();


    for (int iter = 0; iter < max_iter; iter++) {
        //if (idx < 2) {
        //    // printf("Thread ID: %d, Iteration: %d\n", tid, iter); // Debugging line
        //}
        // printf("Thread ID: %d, Iteration: %d\n", tid, iter); // Debugging line
        // Load centroids into shared memory (by threads cooperatively)
       /* for (int i = tid; i < K * image_size; i += blockDim.x) {
            shared_centroids[i] = centroids_d[i];
        }*/



        // Step 1: Assign images to nearest cluster
        if (idx < N) {
            float min_dist = FLT_MAX;
            uint8_t best_cluster = 0;

            for (int k = 0; k < K; ++k) {
                float dist = 0.0f;
                for (int j = 0; j < image_size; ++j) {
                    float diff = static_cast<float>(images_d[idx * image_size + j]) - centroids_d[k * image_size + j];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }
            }

            K_cluster_d[idx] = best_cluster;
        }
        __syncthreads();

        // Step 2: Update centroids (one block handles all clusters)
        if (idx < K) {
            float new_centroid[784] = { 0.0f }; // Assumes max image size
            int count = 0;

            for (int i = 0; i < N; ++i) {
                if (K_cluster_d[i] == idx) {
                    for (int j = 0; j < image_size; ++j) {
                        new_centroid[j] += static_cast<float>(images_d[i * image_size + j]);
                    }
                    count++;
                }
            }

            for (int j = 0; j < image_size; ++j) {
                if (count > 0)
                    centroids_d[idx * image_size + j] = new_centroid[j] / count;
                // else keep previous value
            }
        }
        __syncthreads();
    }
}