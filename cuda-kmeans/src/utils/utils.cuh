#ifndef UTILS_H
#define UTILS_H

#include "cuda_runtime.h"
#include <iostream>

using namespace std;

const int MB = 1048576; // number of bytes in 1 Megabytes
const int KB = 1024; // numbero of bytes in 1 Kilobytes


void printCudaDeviceProperties(cudaDeviceProp& deviceProp);

#endif 