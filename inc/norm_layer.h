/*
 * norm.h
 *
 *  Created on: Nov 24, 2014
 *      Author: manoj
 */

#ifndef NORM_H_
#define NORM_H_
#include <opencv/highgui.h>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;
using namespace cv;

struct norm_layers{
		string name;
		string type;
		vector< double > param;
   };

vector<Mat> norm(vector<Mat> input, vector<double> param);


#endif /* NORM_H_ */
