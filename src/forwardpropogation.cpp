/*
 * forwardpropogation.cpp
 *
 *  Created on: Oct 20, 2014
 *      Author: manoj
 */

#include <opencv/highgui.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <convnet.h>
#include <sigmoid.h>
using namespace std;
using namespace cv;

Convnet CNNFP(Convnet net, Mat training_data)
{
	int nchannels = training_data.channels();
	int rows = training_data.rows - 2*(net.neuronsz/2);
	int cols = training_data.cols - 2*(net.neuronsz/2);
	//layers initializition
	net.layer1.a.clear();
	net.layer2.a.clear();
	net.layer3.a.clear();
	net.layer4.a.clear();
	Mat temp;
	//First Convolution layer
	for(int j = 0; j < net.n_filters; j++)
	{
		Mat temp(rows, cols, CV_64FC(nchannels));
		Mat neuron = net.featuremap1.features[0][j];
		filter2D(training_data, temp, -1, neuron);
		temp = temp(Range(net.neuronsz/2,training_data.rows-net.neuronsz/2), Range(net.neuronsz/2,training_data.cols-net.neuronsz/2));
		add(temp, net.featuremap1.bais[0], temp);
		net.layer1.a.push_back(sigm(temp));
	}


	// First pooling layer
	rows = rows/2;
	cols = cols/2;
	for(int j = 0; j < net.n_filters; j++)
	 {
		Mat kernel = Mat::ones( 2, 2, CV_64F )/ (double)(4);
		filter2D(net.layer1.a[j], temp, -1 , kernel);
		resize(temp, temp, Size(rows, cols), 0.5, 0.5, INTER_NEAREST);
		//cout << temp1.cols << temp1.rows << temp1.channels();
		net.layer2.a.push_back(temp);
	}

	rows = rows - 2*(net.neuronsz/2);
	cols = cols - 2*(net.neuronsz/2);

	//Second Convolution layer
	for(int i = 0; i < net.n_filters; i++)
	{
	  Mat temp = Mat::zeros(rows, cols, CV_64FC(nchannels));
	  for(int j = 0; j < net.n_filters; j++)
	  {
		Mat temp1(rows, cols, CV_64FC(nchannels));
		Mat neuron = net.featuremap2.features[i][j];
		filter2D(net.layer2.a[j], temp1, -1, neuron);
		temp1 = temp1(Range(net.neuronsz/2,net.layer2.a[j].rows-net.neuronsz/2), Range(net.neuronsz/2,net.layer2.a[j].cols-net.neuronsz/2));
		add(temp, temp1, temp);

	  }
	  add(temp, net.featuremap2.bais[i], temp);
	  net.layer3.a.push_back(sigm(temp));
	}

	//Second pooling layer
	rows = rows/2;
	cols = cols/2;
	for(int j = 0; j < net.n_filters; j++)
	{
		Mat kernel = Mat::ones( 2, 2, CV_64F )/ (double)(4);
		filter2D(net.layer3.a[j], temp, -1 , kernel);
		resize(temp, temp, Size(rows, cols), 0.5, 0.5, INTER_NEAREST);
		//cout << temp1.cols << temp1.rows << temp1.channels();
		net.layer4.a.push_back(temp);
	}

	Mat net_a;
	for(uint j = 0; j < net.layer4.a.size(); j++)
	{
		Mat::MSize sa = net.layer4.a[j].size;
		Mat test = net.layer4.a[j];
		test = test.reshape(1, sa[0]*sa[1]);
		net_a.push_back(test);
		//cout << "here" << net_a.cols <<net_a.rows<< net_a.channels();
	}
	Mat rep;
	net.fv = net_a;
	repeat(net.ffb, 1, nchannels, rep);
	net.output = sigm(net.ffw * net_a + rep);
	//cout << "\n\n" <<net.output;
	return net;
}
