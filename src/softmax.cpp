/*
 * softmax.cpp
 *
 *  Created on: Nov 24, 2014
 *      Author: manoj
 */

#include <opencv/highgui.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <softmax.h>
using namespace std;
using namespace cv;

Mat softmax(Mat prob)
{

 double maxVal=0;
 minMaxLoc(prob, 0, &maxVal);
 cout << maxVal;
 subtract(prob, maxVal, prob);
 exp(prob,prob);
 Scalar total = sum(prob);
 divide(prob, total[0], prob);
 return prob;  
 
}



