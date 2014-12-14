/*
 * backwardpropogation.cpp
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
#include <backwardpropogation.h>
using namespace std;
using namespace cv;

Convnet CNNBP(Convnet net, Mat y, Mat x)
{
  Mat temp, temp1;
  int nchannels = x.channels();
  subtract(net.output, y, net.e) ;
  pow(net.e, 2, temp);
  net.L = 0.5 * sum(temp)[0]/nchannels; // Sum gives answer in terms of 4 channels
  //cout << "\n\n"<< net.L;
  //cout << "\n" << net.e;
  Mat te;
  subtract(1, net.output, te);
  multiply(net.output, te, net.od);
  multiply(net.e, net.od, net.od);

  //cout << "\n" << net.od;
  transpose(net.ffw, temp1);
  net.fvd = temp1 * net.od;
  Mat::MSize sa = net.layer4.a[0].size;
  int channels = net.layer4.a[0].channels();
  int fvnum = sa[0] * sa[1];

  // Layer 4
  for(uint j = 0; j < net.layer4.a.size(); j++)
  {
	Mat temp = net.fvd( Range((j)*fvnum, (j+1)*fvnum), Range::all());
	temp = temp.reshape(channels, sa[0]);
	net.layer4.d.push_back(temp);
  }

  //Layer 3
  for(uint j = 0; j < net.layer3.a.size(); j++)
  {
  	Mat temp,temp1;
  	subtract(1,net.layer3.a[j],temp);
  	multiply(net.layer3.a[j], temp, temp);
  	resize(net.layer4.d[j], temp1, Size(0,0), 2, 2, INTER_NEAREST);
  	divide(temp1, 4, temp1);
  	multiply(temp,temp1, temp);
  	net.layer3.d.push_back(temp);
  }

 //Layer 2
for(uint i = 0; i < net.layer2.a.size(); i++)
{
	Mat::MSize sa = net.layer2.a[0].size;
	int channels = net.layer2.a[0].channels();
	Mat z = Mat::zeros(sa[0],sa[1], CV_64FC(channels));
	for(uint j = 0; j < net.layer3.a.size(); j++)
	{
	 Mat neuron = net.featuremap2.features[i][j];
	 flip(neuron,neuron,-1);
	 Mat temp1(sa[0], sa[1], CV_64FC(channels));
 	 filter2D(net.layer3.d[j], temp1, -1, neuron);
 	 resize(temp1, temp1, Size(sa[0], sa[1]), 0.5, 0.5, INTER_NEAREST);
	 add(z, temp1, z);
	}
	net.layer2.d.push_back(z);
}

//Layer 1
  for(uint j = 0; j < net.layer1.a.size(); j++)
  {
  	Mat temp,temp1;
  	subtract(1,net.layer1.a[j],temp);
  	multiply(net.layer1.a[j], temp, temp);
  	resize(net.layer2.d[j], temp1, Size(0,0), 2, 2, INTER_NEAREST);
  	divide(temp1, 4, temp1);
  	multiply(temp,temp1, temp);
  	net.layer1.d.push_back(temp);
  }

//Mixing
  vector<Mat> dk_layer;

  for (uint j = 0; j < net.layer1.a.size(); j++)
  {
	 Mat temp1, temp2, conv_3d;
	 Mat temp = FlipAllDirection(x);
	 Mat::MSize is = temp.size;
	 Mat neuron = net.layer1.d[j];
	 Mat::MSize ns = net.layer1.d[j].size;
	 filter2D(temp, temp1, -1, neuron);
	 temp2 = temp1(Range(ns[0]/2, is[0] - ns[0]/2 + 1), Range(ns[1]/2, is[1] - ns[1]/2+1));

	 // 3D convolution
	 vector<Mat> images;
	 split(temp2, images);
	 conv_3d = Mat::zeros(net.neuronsz,net.neuronsz, CV_64FC1);
	 for(int k =0; k < images.size(); k++)
		 conv_3d = conv_3d + images[k];

	 dk_layer.push_back(conv_3d);
     double a = SumAllChannels(net.layer1.d[j]);
     net.layer1.db.push_back(a);
  }
  net.layer1.dk.push_back(dk_layer);

  //May be problem in dk because of i, j signs
  for(uint j = 0; j < net.layer3.a.size(); j++)
  {
	  vector<Mat> dk_layer;
	  for(int i = 0; i < net.n_filters; i++)
	  {
		  Mat temp1, temp2, conv_3d;
		  Mat temp = FlipAllDirection(net.layer2.a[i]);
		  Mat::MSize is = temp.size;
		  Mat neuron = net.layer3.d[i];
		  Mat::MSize ns = net.layer3.d[i].size;
		  filter2D(temp, temp1, -1, neuron);
		  temp2 = temp1(Range(ns[0]/2, is[0] - ns[0]/2 + 1), Range(ns[1]/2, is[1] - ns[1]/2+1));
		  // 3D convolution
		  vector<Mat> images;
		  split(temp2, images);
		  conv_3d = Mat::zeros(net.neuronsz,net.neuronsz, CV_64FC1);
		  for(int k =0; k < images.size(); k++)
		  	conv_3d = conv_3d + images[k];
		  dk_layer.push_back(conv_3d);
      }
	  net.layer3.dk.push_back(dk_layer);
	  double a = SumAllChannels(net.layer3.d[j]);
	  net.layer3.db.push_back(a);

  }
  Mat trans;
  transpose(net.fv,trans);
  net.dffW = net.od * trans;
  sa = net.od.size;
  net.dffW = net.dffW/sa[1];
  //cout << net.dffW;
  net.dffb = column_mean(net.od);

  return net;
}

Mat column_mean(Mat od)
{
	Mat::MSize sa = od.size;
	Mat ret = Mat::zeros(sa[0], 1, CV_64FC1);
	for(int i = 0; i < sa[0]; i++)
	{
		 Scalar s;
		 s = mean(od(Range(i,i+1), Range::all()));
		 ret.at<double>(i,0) = s[0];
	}
	return ret;
}


double SumAllChannels(Mat X)
{

	Mat temp, mer;
	vector<Mat> images;
	double sum_all;
	int channels;
	split(X, images);
	channels = images.size();
	Scalar s = 0;
    for(int i = 0; i < (channels); i++)
    {
    	s = s + sum(images[0]);
    }

   sum_all = s[0]/channels;
   //cout << sum_all << "\n";
   return sum_all;
}


Mat FlipAllDirection(Mat X)
{

	Mat temp, mer;
	vector<Mat> images;
	int channels;
	flip(X, temp, -1);
	split(temp, images);
	channels = images.size();
    for(int i = 0; i < (channels/2); i++)
    {
      Mat temp = images[i];
      images[i] = images[channels-i-1];
      images[channels-i-1] = temp;
    }

   merge(images, mer);
	return mer;
}

/**Test Cases
 *  //Test of Expand
  int count = 0;
  Mat A = Mat::zeros(2,2,CV_64FC(2));
  for(int k = 0; k < 2; k++)
   for(int i = 0; i < 2; i++ )
	  for(int j = 0; j < 2; j++)
		      A.at<Vec2d>(i,j)[k] = count++;

  //cout << "Earlier" << A << "\n";

  vector<Mat> s;
  split(A,s);
  cout << s[0] << "\n";
  cout << s[1] << "\n";


  Mat B;
  resize(A, B, Size(4,4), 2, 2, INTER_NEAREST);
  //cout << B;
  vector<Mat> s1;
    split(B,s1);
    cout << s1[0];
    cout << s1[1];

 *
 */
