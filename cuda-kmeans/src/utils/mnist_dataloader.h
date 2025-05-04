#ifndef MNIST_DATALOADER_H
#define MNIST_DATALOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

class MNISTDataLoader {
public:
    MNISTDataLoader(const std::string& imagesPath, const std::string& labelsPath);
    void load();
    void visImg(size_t index);
    const std::vector<std::vector<uint8_t>>& getImages() const;
    const std::vector<uint8_t>& getLabels() const;

private:
    std::string imagesPath;
    std::string labelsPath;
    std::vector<std::vector<uint8_t>> images;
    std::vector<uint8_t> labels;

    uint32_t readUint32(std::ifstream& stream);
};

#endif // MNIST_DATALOADER_H
