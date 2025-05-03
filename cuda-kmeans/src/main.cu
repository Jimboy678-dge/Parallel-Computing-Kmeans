#include <iostream>
#include <filesystem>
#include "utils/mnist_dataloader.h"
#include "utils/runner.cuh"
#include "utils/utils.cuh"
#include "kernels/kernels.cuh"

const int imageWidth = 28; // MNIST images are 28x28
const int imageHeight = 28;


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
        uint8_t* images, //flatten images of size N,IMAGE_HEIGHT,IMAGE_WIDTH
        uint8_t N, // number of images = 6000?
        uint8_t IMAGE_HEIGHT, // image height = 28
        uint8_t IMAGE_WIDTH, // image width = 28
        uint8_t* N_cluster, // array to hold N cluster label
        uint8_t K, // k-means parameter
        int seed
    ) {
        kmeans_000000 << <dimGrid, dimBlock >> > (
            images,
            N,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            N_cluster,
            K,
            seed
        );
    }
};

// ADD Extended class here for other kernels


int main() {
    try {
        // Device Properties
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        cout << "Device Count\t" << deviceCount << endl;

        for (int device = 0; device < deviceCount;device++) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, device);
            printCudaDeviceProperties(deviceProp);
        }


        // Load Data
        MNISTDataLoader loader("./data/MNIST/raw/train-images-idx3-ubyte", "./data/MNIST/raw/train-labels-idx1-ubyte");
        loader.load();
        const auto& images = loader.getImages();
        const auto& labels = loader.getLabels();
        std::cout << "Loaded " << images.size() << " images and " << labels.size() << " labels." << std::endl;
        loader.visImg(3456); // visualize one image given data index, comment if you want

        // initialize runner
        K000000Runner runner000000 = K000000Runner();
        //

        // [todo g.agluba]
        // get command-line arguments for easier testing ... 
        //
        runner000000.run(images, images.size(), imageHeight, imageWidth, labels);
        
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
