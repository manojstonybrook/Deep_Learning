/*
 * pooling_layer.h
 *
 *  Created on: Nov 23, 2014
 *      Author: manoj
 */

#ifndef POOLING_LAYER_H_
#define POOLING_LAYER_H_


#include <opencv/cv.h>
#include <string>
#include <vector>
using namespace cv;

struct pool_layers{
		string name;
		string type;
		vector<double> stride;
		vector<double> pad;
	    string method; // image transformation value at each layer
	    vector< double > pool; // matrix 2D 11X11 // channels// total filters
   };

vector<Mat> max_pooler(vector<Mat> input, vector<double> pad,  vector<double> stride, vector<double> pool);


#endif /* POOLING_LAYER_H_ */
