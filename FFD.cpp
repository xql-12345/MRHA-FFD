//Opencv&cuda�Ѳ��ԣ��ɹ�
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

  //  ofstream out3;//�ļ���
 //   out3.open("G://����.txt", ios::app);
 //   out3 << "ffd_match_test ����" << endl;
  ffd_match_test(); 
    //ffd_match_test_Q();
   // Canny();
   // out3 << "bpline_match_test_each ����" << endl;
    bpline_match_test_each();
   
    

}