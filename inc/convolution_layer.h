/*
 * convolution_layer.h
 *
 *  Created on: Nov 23, 2014
 *      Author: manoj
 */

#ifndef CONVOLUTION_LAYER_H_
#define CONVOLUTION_LAYER_H_

#include <opencv/cv.h>
#include <string>
#include <vector>
using namespace cv;

struct conv_layers{
		string name;
		string type;
		vector<double> stride;
		vector<double> pad;
	    vector<float> baises; // bais
	    //vector< vector < vector <Mat> > > filters; // example 11X11 size filter // 3channels// total 96 filters
	    vector< vector <Mat> > filters;
  };


vector<Mat> CNN_FP(vector<Mat> input,  vector< vector <Mat> > filters, vector<float> baises, vector<double> stride, vector<double> pad);
Mat CNN_conv(Mat input,  Mat filter, vector<double>stride, vector<double> pad);

#endif /* CONVOLUTION_LAYER_H_ */
