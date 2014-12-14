/*
 * cnntrain.cpp
 *
 *  Created on: Oct 4, 2014
 *      Author: manoj
 */

#include <convnet.h>
#include<training_data_read.h>
#include <opencv/highgui.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <forwardpropogation.h>
#include <backwardpropogation.h>
#include <applygrads.h>
using namespace std;
using namespace cv;

int main(int argc, char **argv) {

	/** First phase loading image data for training*/
	/*char * training_data = "data/train-images-idx3-ubyte";
	char * labeled_data = "data/train-labels-idx1-ubyte";

	char * testing_data = "data/t10k-images-idx3-ubyte";
    char * testing_label_data = "data/t10k-labels-idx1-ubyte";
	*/
	char * training_data = "data/train_x.bin";
	char * labeled_data = "data/train_y.bin";

	char * testing_data = "data/test_x.bin";
	char * testing_label_data = "data/test_y.bin";
	RNG rng(0);
	int n_rows=28, n_cols=28, number_of_images = 60000, test_images = 10000;
	//read_information(training_data, number_of_images, n_rows, n_cols);
	//read_information(training_data, number_of_images, n_rows, n_cols);

	// Later we will read this dynamically from the net
	int onum = 10;
	double alpha = 1;
	int neuron_size = 5;
	int layers = 2;
	int n_filters = 16;
	Convnet net(neuron_size, layers, n_filters, onum);

	int batchsize = 50;
	int numbatches = number_of_images / batchsize;
	int randno = 0;

	Mat train_data_x(n_rows, n_cols, CV_64FC(batchsize));
	Mat train_data_y(10, batchsize, CV_64FC1);

	//Mat test_data_x(n_rows, n_cols, CV_64FC(batchsize));
	//Mat test_data_y(10, batchsize, CV_64FC1);

	// read_data_cpp(training_data, train_data_x, batchsize, randno);

	double total_time = 0;
	for (int i = 0; i < numbatches; i++)
	{

	   //cout << "HERE1";
	   randno = rng.uniform(0,number_of_images-batchsize);
	   read_data_cpp_match(training_data, labeled_data, train_data_x, train_data_y, batchsize, i*batchsize);
	   double t = (double)getTickCount();
	   net = CNNFP(net, train_data_x);
	   net = CNNBP(net, train_data_y, train_data_x);
	   net = CNNGRADIENTS(net, alpha);
	   t = ((double)getTickCount() - t);
       total_time += t;
       if(i%100 == 0)
	   cout << "\n i="<< i << "time" << total_time/getTickFrequency() << "\n";
		//cout  << train_data_x.at<Vec<double, 50> >(2,2)(0);
	}

	cout << "\n time: "<<total_time/getTickFrequency()<<"\n";
	int error = 0;
	batchsize = 100;
    int test_batches = test_images / batchsize;
	for(int i = 0; i < 1; i++)
	{
	int test_no = 100;
	Mat test_data_x(n_rows, n_cols, CV_64FC(test_no));
	Mat test_data_y(10, test_no, CV_64FC1);
	read_data_cpp_match(testing_data, testing_label_data, test_data_x, test_data_y, test_no, i*test_no);
	net = CNNFP(net, test_data_x);
	Mat out = Mat::zeros(10, test_no, CV_64FC1);
	for(int i = 0; i < test_no; i++)
	{

		double minVal;
		double maxVal;
		Point minLoc;
		Point maxLoc;
		Point maxLoc1;
		//cout << net.output;
		Mat test = net.output(Range(0,10), Range(i,i+1));
		minMaxLoc(test, &minVal, &maxVal, &minLoc, &maxLoc );
		//cout << "\n" << test;
		//cout << "max loc : " << maxLoc << endl;
		//cout << "max val: " << maxVal << endl;
		//cout << i << maxLoc.x << " "<< maxLoc.y << "\n";
		out.at<double>(maxLoc.y, i) = 1;
		Mat res = test_data_y(Range(0,10), Range(i,i+1));
		minMaxLoc( res, &minVal, &maxVal, &minLoc, &maxLoc1 );
		if(maxLoc1 != maxLoc)
			error++;
		//cout << res << " " << out(Range(0,10), Range(i,i+1));
	}
	}
	cout << "\n" <<(double)error;
	cout << "\nFinished\n";
	return 0;

}

/* data read test
 Mat train_data_x1(28, 28, CV_64FC(50));
 Mat train_data_y1(1, 1, CV_64FC(50));
 read_data_cpp(training_data, labeled_data, train_data_x1, train_data_y1, 50, 50);
 cout << "\nHERE1 \n\n" ;
vector<Mat> images;
	    split(train_data_x, images);
	    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
	    for(int i =0; i < 50; i++)
	    {
	     for(int j = 0; j < 10; j++)
	     {

	       if(train_data_y.at<double>(j,i))
	        cout << j << "\n";
	     }
	     imshow( "Display window", images[i] );
	     waitKey(0);

	    }

 }

 */
/*
 * Image show
 for(int k = 0; k < batchsize; k++)
 {
 Mat test(28, 28, CV_8UC1);
 for(int i = 0; i < 28; i ++)
 for(int c = 0; c < 28; c++)
 test.at<uchar>(i,c) = (uchar)(train_data.at<double>(i*28+c+k*28*28));

 namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
 imshow( "Display window", test );
 waitKey(0);
 }
 */

