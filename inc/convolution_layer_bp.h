#ifndef CONVOLUTION_LAYER_BP_H_
#define CONVOLUTION_LAYER_BP_H_

#include <opencv/cv.h>
#include <string>
#include <vector>
using namespace cv;
vector<Mat> CNN_BP(vector<Mat> input,  vector< vector <Mat> > filters, vector<Mat> dzdy, vector<float> baises, vector<double> stride, vector<double> pad);


#endif

