

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>//提供了时间计算的功能函数
#include <iostream>
#include <fstream>//文件流
#include <string>
#include <time.h>
#include <math.h>
#include <windows.h>    //微秒级计时相关函数
#include <random>
#include <ctime>
#include <device_launch_parameters.h>
#include "T.h"
//#include<book.h>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
//#include "gradient_down.h"
//#include <opencv2/xfeatures2d.hpp>


using namespace std;
using namespace cv;



double cal_c(Mat S1, Mat Si, int row, int col)
{
	//互相关小块数目
	int ksize_row = S1.rows / row;
	int ksize_col = S1.cols / col;

	Mat tmp1, tmpi;
	double sum = 0.0;
	for (int i = 0; i < row; i++)
	{
		int i_begin = i * ksize_row;
		for (int j = 0; j < col; j++)
		{
			double sum1 = 0.0;
			double sum2 = 0.0;
			double sum3 = 0.0;
			int j_begin = j * ksize_col;
			for (int t1 = i_begin; t1 < i_begin + ksize_row; t1++)
			{
				uchar* S1_data = S1.ptr<uchar>(t1);
				uchar* Si_data = Si.ptr<uchar>(t1);
				for (int t2 = j_begin; t2 < j_begin + ksize_col; t2++)
				{
					//归一化互相关计算的是图像的灰度值
					sum1 += S1_data[t2] * Si_data[t2];
					sum2 += S1_data[t2] * S1_data[t2];
					sum3 += Si_data[t2] * Si_data[t2];
				}
			}
			sum += sqrt(sum2 * sum3) / (sum1 + 0.0000001);//互相关的倒数
		}
	}
	sum /= (row * col);
	return sum;
}
