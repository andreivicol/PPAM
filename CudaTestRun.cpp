#include <iostream>
#include "./cuda_kernel.cuh"
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/hal/interface.h>
#include <iostream>
#include <string>
#include <lodepng.h>

using namespace cv;
void tvFilter_CPU(const Mat & input, Mat & output);
std::string imageLocation = "goat.png";

int main()
{
    unsigned int imgW, imgH;

    std::vector<unsigned char> img;
    std::vector<unsigned char> outImg;
    lodepng::decode(img, imgW, imgH, imageLocation, LCT_GREY);

    if (0 == img.size()) {
        std::puts("Image could not be loaded! \n");
        return -1;
    }

    Mat input = Mat(imgH, imgW, CV_8UC1);
    Mat output = Mat(imgH, imgW, CV_8UC1);

    memcpy(input.data, img.data(), img.size() * sizeof(unsigned char));

    unsigned char* imgArr = new unsigned char[img.size()];
    unsigned char* outArr = new unsigned char[img.size()];

    int redCnt = 0; int blueCnt = 0; int greenCnt = 0;

    for (auto i = 0; i < img.size(); ++i) {
        imgArr[i] = img.at(i);
    }

    memcpy(outArr, imgArr, img.size());

    kernel(imgArr, outArr, imgW, imgH);

    for (int i = 0; i < img.size(); ++i) {
        outImg.push_back(outArr[i]);
    }

    tvFilter_CPU(input, output);

    std::string outName = "processed" + imageLocation;
    lodepng::encode(outName, outImg, imgW, imgH, LCT_GREY);

    return 0;
}


void tvFilter_CPU(const cv::Mat& input, cv::Mat& output)
{
    Point anchor = Point(-1, -1);
    double delta = 0;
    int ddepth = -1;
    int kernel_size = 3;

    int64 t0 = cv::getTickCount();

    cv::Mat outputi;
    cv::Mat kernel[8];

    kernel[0] = (Mat_<double>(kernel_size, kernel_size) << -1, 0, 0, 0, 1, 0, 0, 0, 0);
    kernel[1] = (Mat_<double>(kernel_size, kernel_size) << 0, -1, 0, 0, 1, 0, 0, 0, 0);
    kernel[2] = (Mat_<double>(kernel_size, kernel_size) << 0, 0, -1, 0, 1, 0, 0, 0, 0);
    kernel[3] = (Mat_<double>(kernel_size, kernel_size) << 0, 0, 0, -1, 1, 0, 0, 0, 0);
    kernel[4] = (Mat_<double>(kernel_size, kernel_size) << 0, 0, 0, 0, 1, -1, 0, 0, 0);
    kernel[5] = (Mat_<double>(kernel_size, kernel_size) << 0, 0, 0, 0, 1, 0, -1, 0, 0);
    kernel[6] = (Mat_<double>(kernel_size, kernel_size) << 0, 0, 0, 0, 1, 0, 0, -1, 0);
    kernel[7] = (Mat_<double>(kernel_size, kernel_size) << 0, 0, 0, 0, 1, 0, 0, 0, -1);

    for (int i = 0; i < 8; i++) {
        // Apply 2D filter
        filter2D(input, outputi, ddepth, kernel[i], anchor, delta, BORDER_DEFAULT);
        output += outputi;
    }

    int64 t1 = cv::getTickCount();
    double secs = (t1 - t0) / cv::getTickFrequency();

    std::cout << "\nProcessing time on CPU (ms): " << secs * 1000 << "\n";
}