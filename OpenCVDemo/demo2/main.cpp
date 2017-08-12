//main.cpp
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"

using namespace std;

int main()
{
    cout << "Hello World!" << endl;

    cv::Mat mat;
    mat = cv::imread("img.JPG");
    cvNamedWindow("hello");
    cv::imshow("hello",mat);

    cvWaitKey(0);

    return 0;
}
