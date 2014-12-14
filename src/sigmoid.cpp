/*
 * sigmoid.cpp
 *
 *  Created on: Oct 18, 2014
 *      Author: manoj
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <opencv/highgui.h>
#include <opencv/cv.h>
using namespace cv;

Mat sigm(Mat P)
{
	multiply(P, -1, P);
	exp(P,P);
	add(P, 1, P);
	divide(1,P, P);
    //double X = 1/(1+exp(-P));
    return P;
}



