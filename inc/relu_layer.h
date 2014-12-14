/*
 * convolution_layer.h
 *
 *  Created on: Nov 23, 2014
 *      Author: manoj
 */

#ifndef RELU_LAYER_H_
#define RELU_LAYER_H_

#include <opencv/cv.h>
#include <string>
#include <vector>
using namespace cv;
vector<Mat> relu(vector<Mat> input);
vector<Mat> relu_bp(vector<Mat> dz, vector<Mat> input);

#endif
