#pragma once
#ifndef _GRADIENT_DOWN_H_
#define _GRADIENT_DOWN_H_

#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;


#define BPLINE_BOARD_SIZE 3


//˫����׼
void bpline_match_test_each(void);
//��һ������׼
void ffd_match_test(void);
//�����׼
void ffd_match_test_Q(void);
void Canny(void);
double cal_cc_block(void);
//void bpline_match_test_each(void);
#endif