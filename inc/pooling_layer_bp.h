/*
 * pooling_layer.h
 *
 *  Created on: Nov 23, 2014
 *      Author: manoj
 */

#ifndef POOLING_LAYER_BP_H_
#define POOLING_LAYER_BP_H_


#include <opencv/cv.h>
#include <string>
#include <vector>
using namespace cv;


vector<Mat> max_pooler_bp(vector<Mat> input, vector<Mat> dzdx, vector<double> pad,  vector<double> stride, vector<double> pool);


#endif /* POOLING_LAYER_H_ */
