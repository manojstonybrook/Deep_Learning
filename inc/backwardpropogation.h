/*
 * backwardpropogation.h
 *
 *  Created on: Oct 20, 2014
 *      Author: manoj
 */

#ifndef BACKWARDPROPOGATION_H_
#define BACKWARDPROPOGATION_H_
#include <opencv/highgui.h>
#include <convnet.h>
using namespace cv;
Convnet CNNBP(Convnet net, Mat label_data, Mat training_data);
Mat FlipAllDirection(Mat X);
double SumAllChannels(Mat X);
Mat column_mean(Mat od);
#endif /* BACKWARDPROPOGATION_H_ */
