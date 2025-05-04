#include "mnist_dataloader.h"
#include "iostream"

MNISTDataLoader::MNISTDataLoader(const std::string& imagesPath, const std::string& labelsPath)
    : imagesPath(imagesPath), labelsPath(labelsPath) {
}

void MNISTDataLoader::load() {
    // Load images
    std::ifstream imagesFile(imagesPath, std::ios::binary);
    if (!imagesFile.is_open()) {
        throw std::runtime_error("Failed to open images file: " + imagesPath);
    }

    uint32_t magicNumber = readUint32(imagesFile);
    if (magicNumber != 2051) {
        throw std::runtime_error("Invalid magic number in images file");
    }

    uint32_t numImages = readUint32(imagesFile);
    uint32_t numRows = readUint32(imagesFile);
    uint32_t numCols = readUint32(imagesFile);

    images.resize(numImages, std::vector<uint8_t>(numRows * numCols));
    for (uint32_t i = 0; i < numImages; ++i) {
        imagesFile.read(reinterpret_cast<char*>(images[i].data()), numRows * numCols);
    }

    // Load labels
    std::ifstream labelsFile(labelsPath, std::ios::binary);
    if (!labelsFile.is_open()) {
        throw std::runtime_error("Failed to open labels file: " + labelsPath);
    }

    magicNumber = readUint32(labelsFile);
    if (magicNumber != 2049) {
        throw std::runtime_error("Invalid magic number in labels file");
    }

    uint32_t numLabels = readUint32(labelsFile);
    labels.resize(numLabels);
    labelsFile.read(reinterpret_cast<char*>(labels.data()), numLabels);
}

const std::vector<std::vector<uint8_t>>& MNISTDataLoader::getImages() const {
    return images;
}

const std::vector<uint8_t>& MNISTDataLoader::getLabels() const {
    return labels;
}

uint32_t MNISTDataLoader::readUint32(std::ifstream& stream) {
   uint32_t value;
   stream.read(reinterpret_cast<char*>(&value), sizeof(value));
   
   // Replace __builtin_bswap32 with a portable implementation
   return ((value & 0xFF000000) >> 24) |
          ((value & 0x00FF0000) >> 8) |
          ((value & 0x0000FF00) << 8) |
          ((value & 0x000000FF) << 24); // Convert from big-endian to little-endian
}

void MNISTDataLoader::visImg(size_t index) {
    // Verify one image
    if (!images.empty() && !labels.empty()) {
        const auto& image = images[index];
        const auto& label = labels[index];

        std::cout << "Verifying image at index " << index << " with label: " << static_cast<int>(label) << std::endl;

        // Display the image as a grid of pixels
        const int imageWidth = 28; // MNIST images are 28x28
        const int imageHeight = 28;

        for (int i = 0; i < imageHeight; ++i) {
            for (int j = 0; j < imageWidth; ++j) {
                // Print pixel intensity (0-255) as a character
                int pixelValue = static_cast<int>(image[i * imageWidth + j]);
                std::cout << (pixelValue > 0 ? "#" : "."); // Threshold for visualization
            }
            std::cout << std::endl;
        }
    }
    else {
        std::cerr << "No images or labels available to verify." << std::endl;
    }
}