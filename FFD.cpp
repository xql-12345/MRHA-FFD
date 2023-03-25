//Opencv&cuda已测试，成功
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "T.h"

//#include "FFD.cu"

//extern "C" void ffd_match_test();
using namespace std;

int main() {

  //  ofstream out3;//文件流
 //   out3.open("G://数据.txt", ios::app);
 //   out3 << "ffd_match_test 方法" << endl;
  ffd_match_test(); 
    //ffd_match_test_Q();
   // Canny();
   // out3 << "bpline_match_test_each 方法" << endl;
    bpline_match_test_each();
   
    

}