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


// [J. Escoto notes]
// The following kernels are from ChatGPT and Copilot, and they are not optimized.
// This kernel uses shared memory to load centroids, but it is not optimized for performance.
// The centroids are loaded into shared memory, but the update step is still inefficient.
// The update step is done in a naive way, which can be improved by using parallel reduction or other techniques.
// ChatGPT and Copilot have provided different implementations, but they all have similar issues.
// The centroid updated in the kernel is not efficient, and it can be improved by using parallel reduction or by using another kernel to update the centroids after the assignment step.
// // from Copilot
// __global__ void kmeans_100000(
//     uint8_t* images_d, // Flattened images of size N X IMAGE_HEIGHT X IMAGE_WIDTH
//     size_t N,          // Number of images
//     uint8_t IMAGE_HEIGHT, // Image height
//     uint8_t IMAGE_WIDTH,  // Image width
//     uint8_t* K_cluster_d, // Array to hold K cluster labels
//     uint8_t K,            // Number of clusters
//     float* centroids_d,   // Flattened centroids of size K X IMAGE_HEIGHT X IMAGE_WIDTH
//     int max_iter          // Maximum number of iterations
// ) {
//     // Thread ID
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     if (idx >= N) return; // Ensure thread does not exceed image count
//     // printf("Thread ID: %d\n", idx); // Debugging line

//     // Image dimensions
//     size_t imageSize = IMAGE_HEIGHT * IMAGE_WIDTH;
//     // printf("Image Size: %zu\n", imageSize); // Debugging line

//     // Shared memory for centroids
//     extern __shared__ float sharedCentroids[];

//     for (int iter = 0; iter < max_iter; ++iter) {
//         // Load centroids into shared memory
//         for (int i = threadIdx.x; i < K * imageSize; i += blockDim.x) {
//             // printf("Loading centroid %d\n", i); // Debugging line
//             sharedCentroids[i] = centroids_d[i];
//             // printf("Centroid %d: %f\n", i, sharedCentroids[i]); // Debugging line
//         }
//         __syncthreads();

//         // Assignment Step: Find the nearest centroid
//         float minDistance = FLT_MAX;
//         uint8_t bestCluster = 0;

//         for (uint8_t cluster = 0; cluster < K; ++cluster) {
//             float distance = 0.0f;
//             for (size_t pixel = 0; pixel < imageSize; ++pixel) {
//                 float diff = images_d[idx * imageSize + pixel] - sharedCentroids[cluster * imageSize + pixel];
//                 distance += diff * diff;
//                 if (cluster < 2 && pixel < 2  && iter < 2 && idx < 2) {
//                     // Debugging line to check the first distance calculation
//                     // cout << "Distance (shared): " << distance << endl; // Debugging line
//                     printf("Distance (shared): %f\n, cluster: %d\n pixel: %d\n, iter: %d\n", distance, cluster, pixel, iter); // Debugging line
//                 }
//             }
//             if (distance < minDistance) {
//                 minDistance = distance;
//                 bestCluster = cluster;
//             }
//         }

//         // Assign the image to the best cluster
//         K_cluster_d[idx] = bestCluster;
//         __syncthreads();

//         // Update Step: Compute new centroids
//         if (threadIdx.x < K) {
//             float newCentroid[784] = { 0 }; // Assuming IMAGE_HEIGHT * IMAGE_WIDTH = 784
//             int clusterSize = 0;

//             for (int i = 0; i < N; ++i) {
//                 if (K_cluster_d[i] == threadIdx.x) {
//                     for (int j = 0; j < imageSize; ++j) {
//                         newCentroid[j] += images_d[i * imageSize + j];
//                     }
//                     clusterSize++;
//                 }
//             }

//             // Average the sum to get the new centroid
//             for (int j = 0; j < imageSize; ++j) {
//                 centroids_d[threadIdx.x * imageSize + j] = clusterSize > 0 ? newCentroid[j] / clusterSize : centroids_d[threadIdx.x * imageSize + j];
//             }
//         }
//         __syncthreads();
//     }
// }


// // from ChatGPT
// __global__ void kmeans_100000(
//     uint8_t* images_d,
//     size_t N,
//     uint8_t IMAGE_HEIGHT,
//     uint8_t IMAGE_WIDTH,
//     uint8_t* K_cluster_d,
//     uint8_t K,
//     float* centroids_d,
//     int max_iter
// ) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     const int IMG_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;

//     extern __shared__ float shared_centroids[];  // shared memory for centroids: size = K * IMG_SIZE

//     for (int iter = 0; iter < max_iter; iter++) {
//         // Step 1: Load centroids into shared memory
//         for (int i = threadIdx.x; i < K * IMG_SIZE; i += blockDim.x) {
//             shared_centroids[i] = centroids_d[i];
//         }
//         __syncthreads();  // Ensure all centroids are loaded

//         // Step 2: Assign clusters
//         if (idx < N) {
//             float minDistance = FLT_MAX;
//             uint8_t bestCluster = 0;

//             for (int k = 0; k < K; k++) {
//                 float distance = 0.0f;
//                 for (int i = 0; i < IMG_SIZE; ++i) {
//                     float diff = images_d[idx * IMG_SIZE + i] - shared_centroids[k * IMG_SIZE + i];
//                     distance += diff * diff;
//                 }
//                 if (distance < minDistance) {
//                     minDistance = distance;
//                     bestCluster = k;
//                 }
//             }

//             K_cluster_d[idx] = bestCluster;
//         }

//         __syncthreads();

//         // Step 3: Update centroids (still inefficient, ideally should be another kernel)
//         if (idx < K) {
//             float newCentroid[784] = { 0 };  // Assumes IMG_SIZE == 784
//             int clusterSize = 0;

//             for (int i = 0; i < N; ++i) {
//                 if (K_cluster_d[i] == idx) {
//                     for (int j = 0; j < IMG_SIZE; ++j) {
//                         newCentroid[j] += images_d[i * IMG_SIZE + j];
//                     }
//                     clusterSize++;
//                 }
//             }

//             for (int j = 0; j < IMG_SIZE; ++j) {
//                 centroids_d[idx * IMG_SIZE + j] = (clusterSize > 0)
//                     ? newCentroid[j] / clusterSize
//                     : centroids_d[idx * IMG_SIZE + j];
//             }
//         }

//         __syncthreads();
//     }
// }

// // from ChatGPT v2
// __global__ void kmeans_100000(
//     uint8_t* images_d,
//     size_t N,
//     uint8_t IMAGE_HEIGHT,
//     uint8_t IMAGE_WIDTH,
//     uint8_t* K_cluster_d,
//     uint8_t K,
//     float* centroids_d,
//     int max_iter
// ) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     const int IMG_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;

//     // Dynamically allocated shared memory for centroids
//     extern __shared__ float shared_centroids[];

//     for (int iter = 0; iter < max_iter; iter++) {
//         // Step 1: Load centroids to shared memory (coalesced)
//         for (int i = threadIdx.x; i < K * IMG_SIZE; i += blockDim.x) {
//             if (i < K * IMG_SIZE) {
//                 shared_centroids[i] = centroids_d[i];
//             }
//         }
//         __syncthreads();

//         // Step 2: Assign image to nearest cluster
//         if (idx < N) {
//             float minDistance = FLT_MAX;
//             uint8_t bestCluster = 0;

//             for (int k = 0; k < K; k++) {
//                 float distance = 0.0f;

//                 for (int i = 0; i < IMG_SIZE; ++i) {
//                     float pixel = static_cast<float>(images_d[idx * IMG_SIZE + i]);
//                     float centroid_val = shared_centroids[k * IMG_SIZE + i];
//                     float diff = pixel - centroid_val;
//                     distance += diff * diff;
//                 }

//                 if (distance < minDistance) {
//                     minDistance = distance;
//                     bestCluster = k;
//                 }
//             }

//             K_cluster_d[idx] = bestCluster;
//         }
//         __syncthreads();

//         // Step 3: Update centroids — inefficient but safe
//         if (idx < K) {
//             float newCentroid[784] = { 0.0f };  // Only safe for IMG_SIZE <= 784
//             int clusterSize = 0;

//             for (int i = 0; i < N; ++i) {
//                 if (K_cluster_d[i] == idx) {
//                     for (int j = 0; j < IMG_SIZE; ++j) {
//                         newCentroid[j] += static_cast<float>(images_d[i * IMG_SIZE + j]);
//                     }
//                     clusterSize++;
//                 }
//             }

//             for (int j = 0; j < IMG_SIZE; ++j) {
//                 centroids_d[idx * IMG_SIZE + j] =
//                     (clusterSize > 0) ? newCentroid[j] / clusterSize
//                                       : centroids_d[idx * IMG_SIZE + j];
//             }
//         }
//         __syncthreads();
//     }
// }


// from ChatGPT v3
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
