#include <opencv2/opencv.hpp>
#include <iostream>
/*

�˴��������ڱ�Ե��⣬Ŀ������ͼͨ����Ե���
�뷨��ͨ����Ե���+FFD������׼��Ԥ����

  

*/
using namespace cv;
using namespace std;

void Canny() {
    Mat src = imread("E://lena.jpg");
    Mat src1 = src.clone();
    Mat dst, edge, gray, edges;

    //��ʼ�����ͼ
    dst.create(src1.size(), src1.type());
    dst = Scalar::all(0);

    //ת�ɻҶ�ͼ
    cvtColor(src1, gray, COLOR_BGR2GRAY);

    //��ֵ�˲����룬Ҳ�����������˲�����
    blur(gray, edges, Size(3, 3));

    //����canny���ӣ��õ���Ե
    Canny(edges, edge, 3, 9, 3);

    //��Ĥ�Ĵ���ʹ��ֻ�б�Ե���ֱ�copy���õ���ɫ�ı�Ե
    src1.copyTo(dst, edge);
    namedWindow("game");
    imshow("Ч��ͼ", dst);
    imwrite("E://lena_1.png", dst);
    waitKey(0);

}