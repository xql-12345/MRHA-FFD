///本代码实现简单的FFD配准，是基于互相关与梯度下降法的FFD配准算法
//已经测试，此程序已经配置opencv、cuda
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
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
//#include "gradient_down.h"
//#include <opencv2/xfeatures2d.hpp>
using namespace std;
using namespace cv;

int c = 1;
int d = 1;
int t1 = 1;
int t2 = 1;
//归一化互相关，测相似度
double cal_cc_block(Mat S1, Mat Si, int row, int col)
{

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


///////////////////////////////////////////////////////互信息度量/////////////////////////////////////////////////////////////////////////////
double Entropy(Mat img)
{
	double temp[256] = { 0.0 };

	// 计算每个像素的累积值
	for (int m = 0; m < img.rows; m++)
	{// 有效访问行列的方式
		const uchar* t = img.ptr<uchar>(m);
		for (int n = 0; n < img.cols; n++)
		{
			int i = t[n];
			temp[i] = temp[i] + 1;
		}
	}

	// 计算每个像素的概率
	for (int i = 0; i < 256; i++)
	{
		temp[i] = temp[i] / (img.rows * img.cols);
	}

	double result = 0;
	// 计算图像信息熵
	for (int i = 0; i < 256; i++)
	{
		if (temp[i] == 0.0)
			result = result;
		else
			result = result - temp[i] * (log(temp[i]) / log(2.0));
	}

	return result;

}


double ComEntropy(Mat img1, Mat img2)
{
	double temp[256][256] = { 0.0 };

	// 计算联合图像像素的累积值
	for (int m1 = 0, m2 = 0; m1 < img1.rows, m2 < img2.rows; m1++, m2++)
	{    // 有效访问行列的方式
		const uchar* t1 = img1.ptr<uchar>(m1);
		const uchar* t2 = img2.ptr<uchar>(m2);
		for (int n1 = 0, n2 = 0; n1 < img1.cols, n2 < img2.cols; n1++, n2++)
		{
			int i = t1[n1], j = t2[n2];
			temp[i][j] = temp[i][j] + 1;
		}
	}

	// 计算每个联合像素的概率
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)

		{
			temp[i][j] = temp[i][j] / (img1.rows * img1.cols);
		}
	}

	double result = 0.0;
	//计算图像联合信息熵
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)

		{
			if (temp[i][j] == 0.0)
				result = result;
			else
				result = result - temp[i][j] * (log(temp[i][j]) / log(2.0));
		}
	}

	//得到两幅图像的互信息熵
	double img1_entropy = Entropy(img1);
	double img2_entropy = Entropy(img2);




	//根据下面代码要求，这个值必须是降低的！

	//互信息
  /*   result = img1_entropy + img2_entropy - result;
   return result;*/
   //NMI归一化互信息的倒数
	/*result =  (img1_entropy + img2_entropy)/result ;
	return result;*/
	 //ECC熵相关系数
	 result = 2*(img1_entropy + img2_entropy - result)/ (img1_entropy + img2_entropy);
	 return 1/result;





}


















//初始化网格
#define randf(a, b) (((rand()%10000+rand()%10000*10000)/100000000.0)*((b)-(a))+(a))
void init_bpline_para(Mat src, int row_block_num, int col_block_num, Mat& grid_points, float min, float max)
{
	int grid_rows = row_block_num + 3;
	int grid_cols = col_block_num + 3;

	int grid_size = grid_rows * grid_cols;
	grid_points.create(Size(2 * grid_size, 1), CV_32FC1);//控制参数的数目：2（row+3）（col+3）


	float* grid_points_data = grid_points.ptr<float>(0);
	srand((unsigned int)time(NULL));//随机数发生器的初始化函数
	for (int i = 0; i < grid_size; i++)
	{
		grid_points_data[i] = randf(min, max);     //x
		grid_points_data[i + grid_size] = randf(min, max);    //y
	}
}


//基于cuda的FFD变换代码
//增加一个参数，对参考图像的变形
__global__ void Bspline_Ffd_kernel(uchar* srcimg, uchar* dstimg, int row_block_num, int col_block_num, float* grid_points, int row, int col)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;  //col
	int y = threadIdx.y + blockDim.y * blockIdx.y;  //row


	if (x < col && y < row)
	{
		int grid_rows = row_block_num + 3;
		int grid_cols = col_block_num + 3;
		int grid_size = grid_rows * grid_cols;
		float delta_x = col * 1.0 / col_block_num;
		float delta_y = row * 1.0 / row_block_num;
		float x_block = x / delta_x;
		float y_block = y / delta_y;

		int j = floor(x_block);
		int i = floor(y_block);
		float u = x_block - j;
		float v = y_block - i;
		float pX[4], pY[4];
		pX[0] = (1 - u * u * u + 3 * u * u - 3 * u) / 6.0;
		pX[1] = (4 + 3 * u * u * u - 6 * u * u) / 6.0;
		pX[2] = (1 - 3 * u * u * u + 3 * u * u + 3 * u) / 6.0;
		pX[3] = u * u * u / 6.0;
		pY[0] = (1 - v * v * v + 3 * v * v - 3 * v) / 6.0;
		pY[1] = (4 + 3 * v * v * v - 6 * v * v) / 6.0;
		pY[2] = (1 - 3 * v * v * v + 3 * v * v + 3 * v) / 6.0;
		pY[3] = v * v * v / 6.0;
		float Tx = 0;
		float Ty = 0;
		for (int m = 0; m < 4; m++)   //行
		{
			for (int n = 0; n < 4; n++)  //列
			{
				int control_point_x = j + n;
				int control_point_y = i + m;


				float temp = pY[m] * pX[n];

				Tx += temp * grid_points[control_point_y * grid_cols + control_point_x];    //x
				Ty += temp * grid_points[control_point_y * grid_cols + control_point_x + grid_size];    //y
			}
		}
		float src_x = x + Tx;
		float src_y = y + Ty;
		int x1 = floor(src_x);
		int y1 = floor(src_y);
		if (x1 < 1 || x1 >= col - 1 || y1 < 1 || y1 >= row - 1)//越界
		{
			dstimg[y * col + x] = 0;
		}
		else
		{
			//dstimg[y*col+x] = srcimg[y1*col+x1];    //最邻近插值
			int x2 = x1 + 1;    //双线性插值
			int y2 = y1 + 1;
			uchar pointa = srcimg[y1 * col + x1];
			uchar pointb = srcimg[y1 * col + x2];
			uchar pointc = srcimg[y2 * col + x1];
			uchar pointd = srcimg[y2 * col + x2];
			uchar gray = (uchar)((x2 - src_x) * (y2 - src_y) * pointa - (x1 - src_x) * (y2 - src_y) * pointb - (x2 - src_x) * (y1 - src_y) * pointc + (x1 - src_x) * (y1 - src_y) * pointd);
			dstimg[y * col + x] = gray;
		}
	}
}

//cuda方法的B样条配准
void Bspline_Ffd_cuda(Mat srcimg, Mat& dstimg, int row_block_num, int col_block_num, Mat grid_points)
{

	dim3 Bpline_Block(16, 16);   //每个线程块有16*16个线程
	int M = (srcimg.cols + Bpline_Block.x - 1) / Bpline_Block.x;
	int N = (srcimg.rows + Bpline_Block.y - 1) / Bpline_Block.y;
	dim3 Bpline_Grid(M, N);


	int grid_rows = row_block_num + 3;
	int grid_cols = col_block_num + 3;
	int grid_size = grid_rows * grid_cols;
	int img_size = srcimg.cols * srcimg.rows;


	uchar* srcimg_cuda;
	uchar* dstimg_cuda;
	float* grid_points_cuda;

	cudaMalloc((void**)&srcimg_cuda, img_size);
	cudaMalloc((void**)&dstimg_cuda, img_size);
	cudaMalloc((void**)&grid_points_cuda, 2 * grid_size * sizeof(float));
	cudaMemcpy(srcimg_cuda, srcimg.data, img_size, cudaMemcpyHostToDevice);
	cudaMemcpy(grid_points_cuda, grid_points.data, 2 * grid_size * sizeof(float), cudaMemcpyHostToDevice);

	Bspline_Ffd_kernel << <Bpline_Grid, Bpline_Block >> > (srcimg_cuda, dstimg_cuda, row_block_num, col_block_num, grid_points_cuda, srcimg.rows, srcimg.cols);


	Mat tmp(srcimg.size(), CV_8UC1);
	cudaMemcpy(tmp.data, dstimg_cuda, img_size, cudaMemcpyDeviceToHost);
	tmp.copyTo(dstimg);


	cudaFree(srcimg_cuda);
	cudaFree(dstimg_cuda);
	cudaFree(grid_points_cuda);
}



//B样条核函数
void Bspline_Ffd(Mat srcimg, Mat& dstimg, int row_block_num, int col_block_num, Mat grid_points)
{
	//让浮动图像规格与参考图像相同
	dstimg.create(srcimg.size(), srcimg.type());
	//获得控制点坐标
	float delta_x = srcimg.cols * 1.0 / col_block_num;
	float delta_y = srcimg.rows * 1.0 / row_block_num;
	int grid_rows = row_block_num + 3;
	int grid_cols = col_block_num + 3;
	int grid_size = grid_rows * grid_cols;

	//B样条控制参数
	float pX[4], pY[4];

	for (int y = 0; y < srcimg.rows; y++)   //B_spline 变形 ，x、y为像素点，遍历
	{
		for (int x = 0; x < srcimg.cols; x++)
		{//取余
			float y_block = y / delta_y;
			float x_block = x / delta_x;
			int i = floor(y_block);
			int j = floor(x_block);
			float u = x_block - j;
			float v = y_block - i;

			//使用基函数计算权重系数
			pX[0] = (1 - u * u * u + 3 * u * u - 3 * u) / 6.0;
			pX[1] = (4 + 3 * u * u * u - 6 * u * u) / 6.0;
			pX[2] = (1 - 3 * u * u * u + 3 * u * u + 3 * u) / 6.0;
			pX[3] = u * u * u / 6.0;
			pY[0] = (1 - v * v * v + 3 * v * v - 3 * v) / 6.0;
			pY[1] = (4 + 3 * v * v * v - 6 * v * v) / 6.0;
			pY[2] = (1 - 3 * v * v * v + 3 * v * v + 3 * v) / 6.0;
			pY[3] = v * v * v / 6.0;


			float Tx = 0;
			float Ty = 0;
			for (int m = 0; m < 4; m++)
			{
				for (int n = 0; n < 4; n++)
				{
					int control_point_x = j + n;
					int control_point_y = i + m;




					float temp = pY[m] * pX[n];
					//含有控制参数的位移
					Tx += temp * grid_points.at<float>(0, control_point_y * grid_cols + control_point_x);    //x
					Ty += temp * grid_points.at<float>(0, control_point_y * grid_cols + control_point_x + grid_size);    //y
				}
			}

			//像素点位移
			float src_x = x + Tx;
			float src_y = y + Ty;
			//向下取整
			int x1 = cvFloor(src_x);
			int y1 = cvFloor(src_y);
			if (x1 < 1 || x1 >= srcimg.cols - 1 || y1 < 1 || y1 >= srcimg.rows - 1)//越界
			{
				dstimg.at<uchar>(y, x) = 0;
			}
			else
			{
				//插法值
				//dstimg.at<uchar>(y, x) = srcimg.at<uchar>(y1, x1);    //最邻近插值
				int x2 = x1 + 1;
				int y2 = y1 + 1;
				uchar pointa = srcimg.at<uchar>(y1, x1);
				uchar pointb = srcimg.at<uchar>(y1, x2);
				uchar pointc = srcimg.at<uchar>(y2, x1);
				uchar pointd = srcimg.at<uchar>(y2, x2);
				uchar gray = (uchar)((x2 - src_x) * (y2 - src_y) * pointa - (x1 - src_x) * (y2 - src_y) * pointb - (x2 - src_x) * (y1 - src_y) * pointc + (x1 - src_x) * (y1 - src_y) * pointd);
				dstimg.at<uchar>(y, x) = gray;
			}
		}
	}
}


//返回两图像相似值
float F_fun_bpline(Mat S1, Mat Si, int row_block_num, int col_block_num, Mat grid_points)
{
	double result;
	Mat Si_tmp;//变形后的浮动图像图像
	//Mat Sl_tem;//变形后的参考图像



	Bspline_Ffd_cuda(Si, Si_tmp, row_block_num, col_block_num, grid_points);
	//Bspline_Ffd(Si, Si_tmp, row_block_num, col_block_num, grid_points);

	//result = cal_cc_block(S1, Si_tmp, 5, 5);//默认分为5*5块计算互相关

	result = ComEntropy(S1, Si_tmp);//默认分为5*5块计算互相关
	return result;
}



//优化
void cal_gradient(Mat S1, Mat Si, int row_block_num, int col_block_num, Mat grid_points, Mat& gradient)
{
	//初始步长
	float EPS = 0.1;

	//创造梯度：梯度为控制的数目
	gradient.create(grid_points.size(), CV_32FC1);
	/*A(参考图像)，B(浮动图像)
	想法：计算的A->B,B->A的相似度a1,a1_1
		  计算每个控制点：梯度改变后，相似度a2,a2_1
		  求导（取最大、最小、中间值等）
	*/
	//计算参考图像与浮动图像的相似度

	float a1 = F_fun_bpline(S1, Si, row_block_num, col_block_num, grid_points);
	float a1_1 = F_fun_bpline(Si, S1, row_block_num, col_block_num, -grid_points);//!!!!!!梯度作用下，参考图像对浮动图像变化,计算相似度


	Mat grid_p = grid_points.clone();
	int t = 0;
	for (int i = 0; i < grid_points.cols; i++)
	{

		grid_p.at<float>(0, i) += EPS;

		float a2_1 = F_fun_bpline(Si, S1, row_block_num, col_block_num, -grid_p);//对参考图像变动后的相似度
		float a2 = F_fun_bpline(S1, Si, row_block_num, col_block_num, grid_p);

		grid_p.at<float>(0, i) -= EPS;


		cout << t++ << endl;
		switch (t1) {
		case 1:gradient.at<float>(0, i) = (a2 - a1) / EPS; break;//原方法
		case 2:gradient.at<float>(0, i) = (((a2 - a1) * 1) - ((a2_1 - a1_1) * 0.1)) / EPS;  break;
		case 3:gradient.at<float>(0, i) = (((a2 - a1) * 1) + ((a2_1 - a1_1) * 0.1)) / EPS; break;
		case 4:gradient.at<float>(0, i) = (((a2 - a1) * 0.9) - ((a2_1 - a1_1) * 0.1)) / EPS;  break;
		case 5:gradient.at<float>(0, i) = min((a2 - a1), (a2_1 - a1_1)) / EPS; break;
		case 6:gradient.at<float>(0, i) = max((a2 - a1), (a2_1 - a1_1)) / EPS; break;
		}

	}
}


//更新
void update_grid_points(Mat& grid_points, Mat gradient, float alp)
{
	for (int i = 0; i < grid_points.cols; i++)
	{
		grid_points.at<float>(0, i) = grid_points.at<float>(0, i) - gradient.at<float>(0, i) * alp;
	}
}


//梯度下降法
int bpline_match(int max_iter, Mat S1, Mat Si, Mat& M, int row_block_num, int col_block_num, Mat& grid_points)//配准图像
{
	//int max_iter = 2000;   //最多迭代次数
	Mat gradient, pre_gradient;
	Mat pre_grid_points;
	double e = 0.00001;//定义迭代精度
	int j = 1;
	//参数初始值
	float alp = 50000;
	float ret1 = 0.0;//上一次迭代的精度
	float ret2 = 0.0;//当前迭代的目标函数值
	int cnt = 0;//迭代计数器

	cal_gradient(S1, Si, row_block_num, col_block_num, grid_points, gradient);//求梯度！！！！
	int out_cnt = 0;
	while (cnt < max_iter)
	{
		//梯度相关
		//cnt++;
		pre_grid_points = grid_points.clone();//原始参数
		update_grid_points(grid_points, gradient, alp);//更新输入参数

		//原始相似度；更新后的相似度
		; ret1 = F_fun_bpline(S1, Si, row_block_num, col_block_num, pre_grid_points);//F_fun(S1, Si, S1_entropy, delta_x, delta_y, pre_grid_points);
		ret2 = F_fun_bpline(S1, Si, row_block_num, col_block_num, grid_points);//F_fun(S1, Si, S1_entropy, delta_x, delta_y, grid_points);

	  //迭代
		if (ret2 > ret1)  //如果当前轮迭代的目标函数值大于上一轮的函数值，则减小步长并重新计算梯度、重新更新参数
		{
			alp *= 0.8;
			grid_points = pre_grid_points.clone();
			continue;
		}

		ofstream out2;//文件流
		out2.open("数据111.txt", ios::app);
		out2 << ret2 << "  " << alp << endl;

		cout << ret2 << "  " << alp << endl;
		//如果变化很小，获得最优值
		if (abs(ret2 - ret1) < e)
		{
			out_cnt++;
			if (out_cnt >= 4)   //如果连续4次目标函数值不改变，则认为求到了最优解，停止迭代
			{
				Bspline_Ffd_cuda(Si, M, row_block_num, col_block_num, grid_points);
				//Bspline_Ffd(Si, M, row_block_num, col_block_num, grid_points);
				return 0;
			}
		}
		else
		{
			out_cnt = 0;
		}
		gradient.copyTo(pre_gradient);
		cal_gradient(S1, Si, row_block_num, col_block_num, grid_points, gradient);//求梯


		if (norm(gradient, NORM_L2) <= norm(pre_gradient, NORM_L2))
			alp *= 3;

		cnt++;
	}
	//return -1;

	Bspline_Ffd_cuda(Si, M, row_block_num, col_block_num, grid_points);
	//Bspline_Ffd(Si, M, row_block_num, col_block_num, grid_points);
	return 0;
}


//FFD旧方法配准
void ffd_match_test(void)
{


	ofstream out2;//文件流
	out2.open("数据111.txt", ios::app);



	Mat img1 = imread("2.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread("1.png", CV_LOAD_IMAGE_GRAYSCALE);
	//在此处确定网格的比例大小，进而确定网格的长宽比例


	while (c <= 2) {







		if (c == 1) {

			while (t1 <= 6) {
				out2 << "这是旧配准" << c << "模式" << t1 << "方法的数据" << endl;


				int row_block_num = 30;
				int col_block_num = 30;



				Mat grid_points;
				Mat grid_points1;
				init_bpline_para(img1, row_block_num, col_block_num, grid_points, -0.00001, 0.00001);
				init_bpline_para(img2, row_block_num, col_block_num, grid_points1, -0.00001, 0.00001);//！！！！给第二个图像套上网格
				//两个图像配准
				Mat out;
				bpline_match(200, img1, img2, out, row_block_num, col_block_num, grid_points);




				int Y = c * 10 + t1;
				char file_img[100];

				sprintf_s(file_img, "OutImage//out3//%d.png", Y);// 给file_img赋值：1.jpg 2.jpg等
				imwrite(file_img, out);


				t1++;
			}
		}
		t1 = 1;
		if (c == 2) {
			while (t1 <= 6) {
				out2 << "这是旧配准" << c << "模式" << t1 << "方法的数据" << endl;
				int k1 = img1.rows;
				int k2 = img1.cols;
				int r = k1 / k2;
				//确定网格数目
				int row_block_num = sqrt(900 * r);
				int col_block_num = sqrt(900 / r);

				Mat grid_points;
				Mat grid_points1;
				init_bpline_para(img1, row_block_num, col_block_num, grid_points, -0.0001, 0.0001);
				init_bpline_para(img2, row_block_num, col_block_num, grid_points1, -0.0001, 0.0001);//！！！！给第二个图像套上网格
				//两个图像配准
				Mat out;
				bpline_match(200, img1, img2, out, row_block_num, col_block_num, grid_points);


				char file_img[100];


				int Y = c * 10 + t1;
				sprintf_s(file_img, "OutImage//out3//%d.png", Y);// 给file_img赋值：1.jpg 2.jpg等
				imwrite(file_img, out);
				  
				t1++;
			}

		}
		c++;

	}



}


//层次配准
void ffd_match_test_Q(void)
{//输入图像


	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);




	Mat img1 = imread("E://lene_1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread("E://lene.jpg", CV_LOAD_IMAGE_GRAYSCALE);




	//确定网格数目
	int row_block_num = 8;
	int col_block_num = 8;
	//初始化控制点
	Mat grid_points;
	init_bpline_para(img1, row_block_num, col_block_num, grid_points, -1, 1);

	//两个图像配准
	Mat out;
	bpline_match(5, img1, img2, out, row_block_num, col_block_num, grid_points);





	//确定网格数目
	row_block_num = 16;
	col_block_num = 16;
	//初始化控制点
  //  Mat grid_points;
	init_bpline_para(img1, row_block_num, col_block_num, grid_points, -1, 1);

	//两个图像配准
	Mat out1;
	bpline_match(5, img1, out, out1, row_block_num, col_block_num, grid_points);

	//确定网格数目
	row_block_num = 30;
	col_block_num = 30;
	//初始化控制点
  //  Mat grid_points;
	init_bpline_para(img1, row_block_num, col_block_num, grid_points, -1, 1);

	//两个图像配准
	Mat out2;
	bpline_match(2000, img1, out1, out2, row_block_num, col_block_num, grid_points);

	namedWindow("game");
	imshow("img1", img1);
	imshow("img2", img2);
	imshow("out", out2);
	imshow("img1-img2", abs(img1 - img2));
	imshow("img1-out", abs(img1 - out));
	imshow("img1-out1", abs(img1 - out1));
	imshow("img1-out2", abs(img1 - out2));

	QueryPerformanceCounter(&t2);
	cout << "time = " << (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart << endl;  //输出时间（单位：ｓ）
	waitKey(0);
}

////////////////////////////////////////////////双向配准实验/////////////////////////////////////////////////////////
//核心就是：整体配准，择优取数
//双向配准：双向配准的初始化
void init_bpline_para_each(Mat src, int row_block_num, int col_block_num, Mat& grid_points_1_i, Mat& grid_points_i_1, float min, float max)
{
	int grid_rows = row_block_num + BPLINE_BOARD_SIZE;
	int grid_cols = col_block_num + BPLINE_BOARD_SIZE;

	int grid_size = grid_rows * grid_cols;

	//定义两种类型的控制点梯度，一种是正方向配准，作用在浮动图像上，另一种是反向配准，作用在固定图像上
	grid_points_1_i.create(Size(2 * grid_size, 1), CV_32FC1);
	grid_points_i_1.create(Size(2 * grid_size, 1), CV_32FC1);


	float* grid_points_data_1_i = grid_points_1_i.ptr<float>(0);
	float* grid_points_data_i_1 = grid_points_i_1.ptr<float>(0);

	srand((unsigned int)time(NULL));//不过为了防止随机数每次重复，常常使用系统时间来初始化
	for (int i = 0; i < grid_size; i++)
	{
		grid_points_data_1_i[i] = randf(min, max);     //x
		grid_points_data_i_1[i] = -grid_points_data_1_i[i];
		grid_points_data_1_i[i + grid_size] = randf(min, max);    //y
		grid_points_data_i_1[i + grid_size] = -grid_points_data_1_i[i + grid_size];

	}

}

//双向配准：梯度下降法
void cal_gradient_each(Mat S1, Mat Si, int row_block_num, int col_block_num, Mat grid_points_1_i, Mat grid_points_i_1, Mat& gradient_1_i, Mat& gradient_i_1)
{
	float EPS = 0.1;//1e-4f;


	gradient_1_i.create(grid_points_1_i.size(), CV_32FC1);
	gradient_i_1.create(gradient_1_i.size(), CV_32FC1);
	//不同之处在于这里面用的并不是反梯度而是真正的配准得来的梯度值
	float a1 = F_fun_bpline(S1, Si, row_block_num, col_block_num, grid_points_1_i);
	float a1_1 = F_fun_bpline(Si, S1, row_block_num, col_block_num, grid_points_i_1);
	//方法的测试
	const float a = 0.8;
	Mat grid_p = grid_points_1_i.clone();
	Mat grid_p_1 = grid_points_i_1.clone();
	int t = 0;
	for (int i = 0; i < grid_points_1_i.cols; i++)
	{
		grid_p.at<float>(0, i) += EPS;
		float a2 = F_fun_bpline(S1, Si, row_block_num, col_block_num, grid_p);
		grid_p.at<float>(0, i) -= EPS;


		grid_p_1.at<float>(0, i) += EPS;
		float a2_1 = F_fun_bpline(Si, S1, row_block_num, col_block_num, grid_p_1);
		grid_p_1.at<float>(0, i) -= EPS;


		cout << t++ << endl;

		switch (t2) {
		case 1:	gradient_1_i.at<float>(0, i) = (a2 - a1) / EPS;										  gradient_i_1.at<float>(0, i) = (a2_1 - a1_1) / EPS; break;//原方法
		case 2:	gradient_1_i.at<float>(0, i) = (((a2 - a1) * 1) - ((a2_1 - a1_1) * 0.1)) / EPS;;	  gradient_i_1.at<float>(0, i) = (((a2_1 - a1_1) * 1) - ((a2 - a1) * 0.1)) / EPS; break;
		case 3:	gradient_1_i.at<float>(0, i) = (((a2 - a1) * 1) + ((a2_1 - a1_1) * 0.1)) / EPS;;		 gradient_i_1.at<float>(0, i) = (((a2_1 - a1_1) * 1) + ((a2 - a1) * 0.1)) / EPS; break;
		case 4:	gradient_1_i.at<float>(0, i) = (((a2 - a1) * 0.9) - ((a2_1 - a1_1) * 0.1)) / EPS;;	 gradient_i_1.at<float>(0, i) = (((a2 - a1) * 0.9) - ((a2_1 - a1_1) * 0.1)) / EPS; break;
		case 5:	gradient_1_i.at<float>(0, i) = min((a2 - a1), (a2_1 - a1_1)) / EPS; ;				 gradient_i_1.at<float>(0, i) = min((a2 - a1), (a2_1 - a1_1)) / EPS; break;
		case 6:	gradient_1_i.at<float>(0, i) = max((a2 - a1), (a2_1 - a1_1)) / EPS; ;				  gradient_i_1.at<float>(0, i) = max((a2 - a1), (a2_1 - a1_1)) / EPS; break;
		}



		//printf("g_1_i=%f, g_i_1=%f\n", g_1_i, g_i_1);

		//gradient_1_i.at<float>(0, i) = (g_1_i + 1.0 / g_i_1)*0.5;
		//gradient_i_1.at<float>(0, i) = (1.0 / g_1_i + g_i_1)*0.5;
		//gradient_1_i.at<float>(0, i) = (g_1_i - g_i_1)*0.5;
		//gradient_i_1.at<float>(0, i) = -gradient_1_i.at<float>(0, i);

		//gradient_1_i.at<float>(0, i) = abs(g_1_i) > abs(g_i_1) ? g_1_i : -g_i_1;
		//gradient_i_1.at<float>(0, i) = abs(g_1_i) > abs(g_i_1) ? -g_1_i : g_i_1;
#if 0
		gradient_1_i.at<float>(0, i) = g_1_i * a - g_i_1 * (1 - a);
		gradient_i_1.at<float>(0, i) = g_i_1 * a - g_1_i * (1 - a);
		//gradient_1_i.at<float>(0, i) = g_1_i - g_i_1*a;
		//gradient_i_1.at<float>(0, i) = g_i_1 - g_1_i*a;
#else

		//gradient_1_i.at<float>(0, i) = g_1_i;
		//gradient_i_1.at<float>(0, i) = g_i_1;
#endif
	}
}


//双向配准：根据梯度更新代码
void update_grid_points_each(Mat& grid_points_1_i, Mat& grid_points_i_1, Mat gradient_1_i, Mat gradient_i_1, float alpha)
{
	//const float a = 0.5;
	for (int i = 0; i < grid_points_1_i.cols; i++)
	{
		//grid_points_1_i.at<float>(0, i) = grid_points_1_i.at<float>(0, i) - gradient_1_i.at<float>(0, i)*alpha;
		//grid_points_i_1.at<float>(0, i) = grid_points_i_1.at<float>(0, i) - gradient_i_1.at<float>(0, i)*alpha;

		grid_points_1_i.at<float>(0, i) = grid_points_1_i.at<float>(0, i) - gradient_1_i.at<float>(0, i) * alpha;
		grid_points_i_1.at<float>(0, i) = grid_points_i_1.at<float>(0, i) - gradient_i_1.at<float>(0, i) * alpha;


		//grid_points_1_i.at<float>(0, i) = grid_points_1_i.at<float>(0, i)*a - grid_points_i_1.at<float>(0, i)*(1 - a);
		//grid_points_i_1.at<float>(0, i) = grid_points_i_1.at<float>(0, i)*a - grid_points_1_i.at<float>(0, i)*(1 - a);
	}
}


//双向配准：FFD+梯度下降法配准
int bpline_match_each(Mat S1, Mat Si, Mat& M, int row_block_num, int col_block_num, Mat& grid_points_1_i, Mat& grid_points_i_1)
{
	int max_iter = 5000;   //最多迭代次数
	Mat gradient_1_i, gradient_i_1;
	Mat pre_grid_points_1_i, pre_grid_points_i_1;
	double e = 0.00001;//定义迭代精度
	float ret1 = 0.0;
	float ret2 = 0.0;
	float ret3 = 0.0;
	float ret4 = 0.0;
	int cnt = 0;
	float alpha = 50000;

	//cal_gradient(S1, Si, row_block_num, col_block_num, grid_points, gradient);   //求梯度
	cal_gradient_each(S1, Si, row_block_num, col_block_num, grid_points_1_i, grid_points_i_1, gradient_1_i, gradient_i_1);
	int out_cnt = 0;

	Mat pre_gradient_1_i = Mat::zeros(grid_points_1_i.size(), CV_32FC1);
	Mat pre_gradient_i_1 = Mat::zeros(grid_points_1_i.size(), CV_32FC1);



	while (cnt < max_iter)
	{


		pre_grid_points_1_i = grid_points_1_i.clone();
		pre_grid_points_i_1 = grid_points_i_1.clone();

		update_grid_points_each(grid_points_1_i, grid_points_i_1, gradient_1_i, gradient_i_1, alpha);  //更新输入参数

		ret1 = F_fun_bpline(S1, Si, row_block_num, col_block_num, pre_grid_points_1_i);
		ret2 = F_fun_bpline(S1, Si, row_block_num, col_block_num, grid_points_1_i);

#if 1
		ret3 = F_fun_bpline(Si, S1, row_block_num, col_block_num, pre_grid_points_i_1);
		ret4 = F_fun_bpline(Si, S1, row_block_num, col_block_num, grid_points_i_1);

		//	printf("ret1=%f, ret2=%f, ret3=%f, ret4=%f, α=%f\n", ret1, ret2, ret3, ret4, alpha);



		if (ret2 > ret1 || ret4 > ret3)  //如果当前轮迭代的目标函数值大于上一轮的函数值，则减小步长并重新计算梯度、重新更新参数
		{
			alpha *= 0.8;
			pre_grid_points_1_i.copyTo(grid_points_1_i);
			pre_grid_points_i_1.copyTo(grid_points_i_1);
			continue;
		}

#if 1
		if (ret2 > ret4)
		{
			grid_points_1_i = -grid_points_i_1;
		}
		else if (ret4 > ret2)
		{
			grid_points_i_1 = -grid_points_1_i;
		}
#endif
		ofstream out3;//文件流
		out3.open("数据T111.txt", ios::app);
		out3 <<min(ret2,ret4)<< "  " << alpha << endl;
		cout << min(ret2, ret4) << "  " << alpha << endl;
		if (abs(ret2 - ret1) < e)
		{
			out_cnt++;
			if (out_cnt >= 4)   //如果连续4次目标函数值不改变，则认为求到了最优解，停止迭代
			{
				if (ret2 <= ret4)
				{
					Bspline_Ffd_cuda(Si, M, row_block_num, col_block_num, grid_points_1_i);
				}
				else
				{
					Bspline_Ffd_cuda(Si, M, row_block_num, col_block_num, -grid_points_i_1);
				}


				return 0;
			}
		}
		else
		{
			out_cnt = 0;
		}
#else
		//	printf("ret1=%f, ret2=%f, α=%f\n", ret1, ret2, alpha);


		if (ret2 > ret1)  //如果当前轮迭代的目标函数值大于上一轮的函数值，则减小步长并重新计算梯度、重新更新参数
		{
			alpha *= 0.8;
			pre_grid_points_1_i.copyTo(grid_points_1_i);
			pre_grid_points_i_1.copyTo(grid_points_i_1);
			continue;
		}

		if (abs(ret2 - ret1) < e)
		{
			out_cnt++;
			if (out_cnt >= 4)   //如果连续4次目标函数值不改变，则认为求到了最优解，停止迭代
			{

				Bspline_Ffd_cuda(Si, M, row_block_num, col_block_num, grid_points_1_i);


				return 0;
			}
		}
		else
		{
			out_cnt = 0;
		}

#endif

		gradient_1_i.copyTo(pre_gradient_1_i);
		gradient_i_1.copyTo(pre_gradient_i_1);
		cal_gradient_each(S1, Si, row_block_num, col_block_num, grid_points_1_i, grid_points_i_1, gradient_1_i, gradient_i_1);  //求梯度

		//if (norm(gradient_1_i, NORM_L2) <= norm(pre_gradient_1_i, NORM_L2) || norm(gradient_i_1, NORM_L2) <= norm(pre_gradient_i_1, NORM_L2))  //如果梯度相比上一次迭代有所下降，则增大步长
		if (norm(gradient_1_i, NORM_L2) <= norm(pre_gradient_1_i, NORM_L2))
			alpha *= 3;

		cnt++;
	}


	return -1;


}

//双向配准：配准主函数
void bpline_match_test_each(void)
{

	ofstream out3;//文件流
	out3.open("数据T111.txt", ios::app);
	Mat img1 = imread("2.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread("1.png", CV_LOAD_IMAGE_GRAYSCALE);

	while (d <= 2) {



		if (d == 1) {
			while (t2 <= 6) {


				out3 << "这是新配准" << d << "模式" << t2 << "方法的数据" << endl;

				int row_block_num = 30;
				int col_block_num = 30;


				Mat grid_points_1_i, grid_points_i_1;
				//init_bpline_para(img1, row_block_num, col_block_num, grid_points, -0.01, 0.01);
				init_bpline_para_each(img1, row_block_num, col_block_num, grid_points_1_i, grid_points_i_1, -0.0001, 0.0001);
				Mat out;
				//bpline_match(img1, img2, out, row_block_num, col_block_num, grid_points);
				//level_ffd_match(img1, img2, out);
				bpline_match_each(img1, img2, out, row_block_num, col_block_num, grid_points_1_i, grid_points_i_1);


				char file_img[100];


				int Y = d * 100 + t2;
				sprintf_s(file_img, "OutImage//out3//%d.png", Y);// 给file_img赋值：1.jpg 2.jpg等
				imwrite(file_img, out);






				t2++;
			}
		}
		t2 = 1;
		if (d == 2) {
			while (t2 <= 6) {
				out3 << "这是新配准" << d << "模式" << t2 << "方法的数据" << endl;

				int k1 = img1.rows;//图像长
				int k2 = img1.cols; // 图像宽
				int r = k1 / k2;//比值
				int row_block_num = sqrt(900 * r);//行网格数
				int col_block_num = sqrt(900 / r);//列网格数


				Mat grid_points_1_i, grid_points_i_1;
				//init_bpline_para(img1, row_block_num, col_block_num, grid_points, -0.01, 0.01);
				init_bpline_para_each(img1, row_block_num, col_block_num, grid_points_1_i, grid_points_i_1, -0.0001, 0.0001);
				Mat out;
				//bpline_match(img1, img2, out, row_block_num, col_block_num, grid_points);
				//level_ffd_match(img1, img2, out);
				bpline_match_each(img1, img2, out, row_block_num, col_block_num, grid_points_1_i, grid_points_i_1);


				char file_img[100];



				int Y = d * 100 + t2;
				sprintf_s(file_img, "OutImage//out3//%d.png", Y);// 给file_img赋值：1.jpg 2.jpg等
				imwrite(file_img, out);







				t2++;
			}
		}

		d++;
	}

}
