#include <opencv2/opencv.hpp>
#include <iostream>
/*

此代码是用于边缘检测，目的是试图通过边缘检测
想法：通过边缘检测+FFD进行配准的预处理

  

*/
using namespace cv;
using namespace std;

void Canny() {
    Mat src = imread("E://lena.jpg");
    Mat src1 = src.clone();
    Mat dst, edge, gray, edges;

    //初始化输出图
    dst.create(src1.size(), src1.type());
    dst = Scalar::all(0);

    //转成灰度图
    cvtColor(src1, gray, COLOR_BGR2GRAY);

    //均值滤波降噪，也可以用其他滤波方法
    blur(gray, edges, Size(3, 3));

    //运行canny算子，得到边缘
    Canny(edges, edge, 3, 9, 3);

    //掩膜的存在使得只有边缘部分被copy，得到彩色的边缘
    src1.copyTo(dst, edge);
    namedWindow("game");
    imshow("效果图", dst);
    imwrite("E://lena_1.png", dst);
    waitKey(0);

}