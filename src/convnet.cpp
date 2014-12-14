/*
 * convnet.cpp
 *
 *  Created on: Oct 4, 2014
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

Convnet::Convnet()
{

}

Convnet::Convnet(int neuron_size, int layers, int filters, int output)
{
	neuronsz =  neuron_size;
	layersz = layers;
    n_filters = filters;
    n_output = output;
    fvnum = filters * filters;
    feature_init();
}

void Convnet::feature_init()
{
	// Hard code initialization we can optimize it at this point of time it is for testing
	RNG rng(0);
	int init_size = n_filters * neuronsz * neuronsz;
	featuremap1.feature_size = init_size;
	featuremap1.id = 0;

// We can add all the features in the vectors at this point hard coded to see results Layer 1

	vector<Mat> weight;
	for(int i = 0; i < n_filters; i++)
	{
		 Mat w(neuronsz,neuronsz, CV_64FC1);

		for(int r = 0; r < neuronsz; r++)
			for(int c = 0; c < neuronsz; c++)
			{
				/*double a;
				if(r%2==0)
					a = 0.1;
				else
					a = -0.1;*/
				double a = ((double)(rng.uniform(0,100))/100 - 0.5) * 0.001;
				w.at<double>(r,c) = a;
			}
		weight.push_back(w);
	}


  featuremap1.features.push_back(weight);

   for(int j = 0; j < n_filters; j++)
   {
	   vector<Mat>  weight;
	   for(int i = 0; i < n_filters; i++)
		{
		    Mat w(neuronsz,neuronsz, CV_64FC1);

			for(int r = 0; r < neuronsz; r++)
				for(int c = 0; c < neuronsz; c++)
				{
					/*double a;
					if(r%2==0)
						a = 0.1;
					else
						a = -0.1;
					*/
					double a = ((double)(rng.uniform(0,100))/100 - 0.5) * 0.001;
					w.at<double>(r,c) = a;

				}
			weight.push_back(w);
		}
	 featuremap2.features.push_back(weight);
	}

    //layer2
    init_size  = init_size * n_filters;
	featuremap2.feature_size =  init_size;
	featuremap2.id = 1;
	for(int i = 0; i < n_filters; i++)
	{
		featuremap1.bais.push_back(0);
		featuremap2.bais.push_back(0);
	}
	//Output layer
	ffw = Mat::zeros(n_output, fvnum, CV_64FC1);
    for(int i = 0; i < n_output; i++)
    {
    	for(int j =0; j < fvnum; j++)
    	{
    		double a = ((double)(rng.uniform(0,100))/100 - 0.5) * 0.001;
    		ffw.at<double>(i,j) = a;
    		//ffw.at<double>(i,j) = 0;
    	}

    }
    ffb = Mat::zeros(n_output,1,CV_64FC1);
    dffb = Mat::zeros(n_output,1,CV_64FC1);
    e = Mat::zeros(1, 50,CV_64FC1 );

}

Convnet::~Convnet()
{

}


/*
void Convnet::cnnfp(Mat training_data)
{
 	int nchannels = training_data.channels();
 	int rows = training_data.rows - 2*(neuronsz/2);
 	int cols = training_data.cols - 2*(neuronsz/2);
 	//layers initializition
 	layer1.a.clear();
 	layer2.a.clear();
 	layer3.a.clear();
 	layer4.a.clear();
    Mat temp;
 	//First Convolution layer
 	for(int j = 0; j < n_filters; j++)
 	{
 		Mat temp(rows, cols, CV_64FC(nchannels));
 		Mat neuron = featuremap1.features[0][j];
 		filter2D(training_data, temp, -1, neuron);
 		temp = temp(Range(neuronsz/2,training_data.rows-neuronsz/2), Range(neuronsz/2,training_data.cols-neuronsz/2));
 		add(temp, featuremap1.bais[0], temp);
 		layer1.a.push_back(sigm(temp));
 	}


 	// First pooling layer
 	rows = rows/2;
 	cols = cols/2;
 	for(int j = 0; j < n_filters; j++)
 	 {
 		Mat kernel = Mat::ones( 2, 2, CV_64F )/ (double)(4);
 		filter2D(layer1.a[j], temp, -1 , kernel);
 		resize(temp, temp, Size(rows, cols), 0.5, 0.5, INTER_NEAREST);
 		//cout << temp1.cols << temp1.rows << temp1.channels();
 		layer2.a.push_back(temp);
    }

 	rows = rows - 2*(neuronsz/2);
 	cols = cols - 2*(neuronsz/2);

 	//Second Convolution layer
 	for(int i = 0; i < n_filters; i++)
 	{
 	  Mat temp = Mat::zeros(rows, cols, CV_64FC(nchannels));
 	  for(int j = 0; j < n_filters; j++)
 	  {
 		Mat temp1(rows, cols, CV_64FC(nchannels));
		Mat neuron = featuremap2.features[i][j];
		filter2D(layer2.a[j], temp1, -1, neuron);
		temp1 = temp1(Range(neuronsz/2,layer2.a[j].rows-neuronsz/2), Range(neuronsz/2,layer2.a[j].cols-neuronsz/2));
		add(temp, temp1, temp);

 	  }
 	  add(temp, featuremap2.bais[i], temp);
 	  layer3.a.push_back(sigm(temp));
 	}

 	//Second pooling layer
 	rows = rows/2;
	cols = cols/2;
	for(int j = 0; j < n_filters; j++)
	{
		Mat kernel = Mat::ones( 2, 2, CV_64F )/ (double)(4);
		filter2D(layer3.a[j], temp, -1 , kernel);
		resize(temp, temp, Size(rows, cols), 0.5, 0.5, INTER_NEAREST);
		//cout << temp1.cols << temp1.rows << temp1.channels();
		layer4.a.push_back(temp);
	}

	Mat net_a;
	for(int j = 0; j < layer4.a.size(); j++)
	{
		Mat::MSize sa = layer4.a[j].size;
		Mat test = layer4.a[j];
		test = test.reshape(1, sa[0]*sa[1]);
		net_a.push_back(test);
		//cout << "here" << net_a.cols <<net_a.rows<< net_a.channels();
	}


	Mat output = sigm(ffw * net_a);
	// cout << "\n\n"<< output;
	/*
	for(int i = 0; i < 10; i++)
	{
	 if(output.at<double>(i,0) != output.at<double>(i,49))
	  cout << " "<<output.at<double>(i,0);
	  //cout << " "<<output.at<double>(i,10);
	 cout << "\n ";
	}
	cout << " \n";
	*/

//}


/* Test Cases
 *
 * 		vector<Mat> images;
 		split(temp, images);
 		namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
 		imshow( "Display window", images[0] );
 		waitKey(0);
 		cout << temp.cols << " "<< temp.channels() << " " << training_data.at<Vec<double, 50> >(2,2)(0) << " "<<temp.at<Vec<double, 50> >(2,2)(0);
 *
 *
 * Interpolation test
 *int count =0;
 	Mat inter(4,4, CV_64FC1);
 	Mat inter1;
 	for(int i = 0; i < 4; i++)
 		for(int j = 0; j < 4; j++)
 			inter.at<double>(i,j) = count++;
 	cout  << "\n";
 	resize(inter, inter1, Size(2, 2), 0.5, 0.5, INTER_NEAREST);
 	cout << inter1.cols << inter1.rows;
 	for(int i = 0; i < 2; i++)
 	{
 	 		for(int j = 0; j < 2; j++)
 	 			cout << " "<<inter1.at<double>(i,j);
 	 		cout << "\n";
 	}
 */

