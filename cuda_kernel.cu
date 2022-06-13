#include "cuda_runtime.h"
#include "./cuda_kernel.cuh"
#include <iostream>
#include <opencv2/core/core.hpp>


const int FILTER_WIDTH = 3;
const int FILTER_HEIGHT = 3;
const int BLOCK_SIZE = 16;

__global__ void totalVarFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
    {
        float sod = 0;
        for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++) {
            for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++) {
                float fl = srcImage[((y + ky) * width + (x + kx))];
                float center = srcImage[((y)*width + (x))];
                sod += fl - center;
            }
        }
        dstImage[(y * width + x)] = sod;
    }
}



void kernel(unsigned char* srcImage, unsigned char* destImg, unsigned int width, unsigned int height) {
    unsigned char* d_srcImage, *d_dstImage;

    cudaMalloc((void**)&d_srcImage, width * height);
    cudaMalloc((void**)&d_dstImage, width * height);

    cudaMemcpy(d_srcImage, srcImage, width * height, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_dstImage, srcImage, width * height, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    auto kernelT0 = cv::getTickCount();
    totalVarFilter << <grid, block >> > (d_srcImage, d_dstImage, width, height);
    auto kernelDelta = (cv::getTickCount() - kernelT0) / cv::getTickFrequency() * 1000.0000f;
    std::cout << "Time taken by kernel (ms): " << kernelDelta << '\n';

    cudaMemcpy(destImg, d_dstImage, width * height, cudaMemcpyDeviceToHost);
}
