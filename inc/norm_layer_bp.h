#ifndef NORM_BP_H_
#define NORM_BP_H_
#include <opencv/highgui.h>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;
using namespace cv;

vector<Mat> norm_bp(vector<Mat> input, vector<double> param, vector<Mat> dzdy);
#endif
