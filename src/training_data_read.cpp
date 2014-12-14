/*
 * training_data_read.cpp
 *
 *  Created on: Oct 4, 2014
 *      Author: manoj
 */
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <convolution_layer.h>
#include <pooling_layer.h>
#include <norm_layer.h>
using namespace std;
using namespace cv;

void read_conv_layer(char * input, conv_layers  *layer)
{
	ifstream f1(input,ios::in | ios::binary); // image data

	if (!f1.is_open())
	{
		cerr << "ERROR: Can't open conv.bin files. Please locate them in current directory" << endl;
		return;
	}
	int stride_size=0;  //x and y
	f1.read(reinterpret_cast<char*>(&stride_size),sizeof(stride_size));
	//cout << stride_size;
	double stride_x, stride_y;
	f1.read(reinterpret_cast<char*>(&stride_x),sizeof(stride_x));
    //cout << "\n"<<stride_x;
    f1.read(reinterpret_cast<char*>(&stride_y),sizeof(stride_y));
    //cout << "\n"<<stride_y;
    layer->stride.push_back(stride_x);
    layer->stride.push_back(stride_y);

    //padding data
    int pad_size;
    f1.read(reinterpret_cast<char*>(&pad_size),sizeof(pad_size));
    //cout << pad_size;
    for(int i = 0; i < pad_size; i++)
    {
    	double pad;
    	f1.read(reinterpret_cast<char*>(&pad),sizeof(pad));
        layer->pad.push_back(pad);
    }

    //baising data
    int bais_size;
	f1.read(reinterpret_cast<char*>(&bais_size),sizeof(bais_size));
	//cout << bais_size;
	for(int i = 0; i < bais_size; i++)
	{
	    float bais;
		f1.read(reinterpret_cast<char*>(&bais),sizeof(bais));
		//cout << bais;
		layer->baises.push_back(bais);
	}

	//filter data
	int row, col, channels, filters;
	f1.read(reinterpret_cast<char*>(&row),sizeof(row));
	f1.read(reinterpret_cast<char*>(&col),sizeof(col));
	f1.read(reinterpret_cast<char*>(&channels),sizeof(channels));
	f1.read(reinterpret_cast<char*>(&filters),sizeof(filters));
	for(int f = 0; f < filters; f++)
	{
	  vector<Mat> RGB;
	  for(int c = 0; c < channels; c++)
	  {
		  Mat single_channel(row, col, CV_64FC1);

		   for(int c = 0; c < col; c++)
			for(int r = 0; r < row; r++)
			{
				float temp;
				f1.read(reinterpret_cast<char*>(&temp),sizeof(temp));
				single_channel.at<double>(r,c) = (double)temp;
			}
		RGB.push_back(single_channel);

	  }
		layer->filters.push_back(RGB);
	}


	//cout << "\n"<< row << col << channels << filters;


}


void read_pool_layer(char * input, pool_layers  *layer)
{
	ifstream f1(input,ios::in | ios::binary); // image data

	if (!f1.is_open())
	{
		cerr << "ERROR: Can't open pool.bin files. Please locate them in current directory" << endl;
		return;
	}
	int stride_size=0;  //x and y
	f1.read(reinterpret_cast<char*>(&stride_size),sizeof(stride_size));
	//cout << stride_size;
	double stride_x, stride_y;
	f1.read(reinterpret_cast<char*>(&stride_x),sizeof(stride_x));
    //cout << "\n"<<stride_x;
    f1.read(reinterpret_cast<char*>(&stride_y),sizeof(stride_y));
    //cout << "\n"<<stride_y;
    layer->stride.push_back(stride_x);
    layer->stride.push_back(stride_y);

    //padding data
    int pad_size;
    f1.read(reinterpret_cast<char*>(&pad_size),sizeof(pad_size));
    //cout << pad_size;
    for(int i = 0; i < pad_size; i++)
    {
    	double pad;
    	f1.read(reinterpret_cast<char*>(&pad),sizeof(pad));
        layer->pad.push_back(pad);
    }

    //pooling data
	int pool_size;
	f1.read(reinterpret_cast<char*>(&pool_size),sizeof(pool_size));
	//cout << pad_size;
	for(int i = 0; i < pool_size; i++)
	{
		double pool;
		f1.read(reinterpret_cast<char*>(&pool),sizeof(pool));
		layer->pool.push_back(pool);
		//cout << pool;
	}

}

void read_norm_layer(char * input, norm_layers  *layer)
{
	ifstream f1(input,ios::in | ios::binary); // image data

	if (!f1.is_open())
	{
		cerr << "ERROR: Can't open norm.bin files. Please locate them in current directory" << endl;
		return;
	}

    //param data
    int param_size;
    f1.read(reinterpret_cast<char*>(&param_size),sizeof(param_size));
    //cout << pad_size;
    for(int i = 0; i < param_size; i++)
    {
    	double param;
    	f1.read(reinterpret_cast<char*>(&param),sizeof(param));
        layer->param.push_back(param);
        //cout << "\n"<<param  ;
    }
}


void read_normalize(char * input, Mat data, int width, int height, int nchannels)
{

	    // Represent MNIST datafiles as C++ file streams f1 and f2 respectively
		ifstream f1(input,ios::in | ios::binary); // image data

		if (!f1.is_open())
		{
			cerr << "ERROR: Can't open normalize.bin files. Please locate them in current directory" << endl;
			return;
		}
		// Create buffers for image data and correct labels

		for(int n = 0; n < nchannels; n++)
		{
			for(int r=0;r < width; ++r)
		    {
			  for(int c=0; c < height; ++c)
			  {
				double temp = 0;
				f1.read(reinterpret_cast<char*>(&temp),sizeof(temp));
				data.at<Vec<double, 3> >(c,r)(n)=(double)temp;
			  }
		    }

		}
}

