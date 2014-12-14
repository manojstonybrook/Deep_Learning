/*
 * applygrads.cpp
 *
 *  Created on: Oct 25, 2014
 *      Author: manoj
 */

#include <opencv/highgui.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <convnet.h>
#include <sigmoid.h>
#include <applygrads.h>
using namespace std;
using namespace cv;

Convnet CNNGRADIENTS(Convnet net, double alpha)
{

	for(int j = 0; j < net.n_filters; j++)
	{
		Mat test;
		multiply(net.layer1.dk[0][j], alpha, test);
		subtract(net.featuremap1.features[0][j], test, net.featuremap1.features[0][j]);
		multiply(net.layer1.db[j], alpha, test);
		net.featuremap1.bais[j] = net.featuremap1.bais[j] - alpha *net.layer1.db[j];
	}

	for(int j = 0; j < net.n_filters; j++)
	{
		for(int i = 0; i < net.n_filters; i++)
		{
			Mat test;
			multiply(net.layer3.dk[i][j], alpha, test);
			subtract(net.featuremap2.features[i][j], test, net.featuremap2.features[i][j]);
		}
		net.featuremap2.bais[j] =  net.featuremap2.bais[j] - alpha*net.layer3.db[j];
	}

	Mat temp;
	multiply(net.dffW, alpha, temp);
	net.ffw = net.ffw - temp;
	multiply(net.dffb, alpha, temp);
	net.ffb = net.ffb - temp;

	return net;
}

