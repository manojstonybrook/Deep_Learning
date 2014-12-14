/*
 * forwardpropogation.h
 *
 *  Created on: Oct 20, 2014
 *      Author: manoj
 */

#ifndef FORWARDPROPOGATION_H_
#define FORWARDPROPOGATION_H_

#include <opencv/highgui.h>
#include <convnet.h>
using namespace cv;

Convnet CNNFP(Convnet net, Mat training_data);


#endif /* FORWARDPROPOGATION_H_ */
