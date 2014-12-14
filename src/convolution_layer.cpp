/*
 * convolution_layer.cpp
 *
 *  Created on: Nov 23, 2014
 *      Author: manoj
 */

#include <opencv/highgui.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sigmoid.h>
#include <convolution_layer.h>
using namespace std;
using namespace cv;

vector<Mat> CNN_FP(vector<Mat> input,  vector< vector <Mat> > filters, vector<float> baises, vector<double> stride, vector<double> pad)
{

  vector<Mat> ret;
 	 
  int group = input.size()/filters[0].size();
  int in_g = input.size()/group;
  int out_g = filters.size()/group;
 
  for(int g = 1; g <= group; g++)
  {
	  for(int i = out_g*(g-1); i < out_g*g; i++ )
	  {
		  int filter_index = 0;
		  Mat temp1 = Mat::zeros((input[0].rows + (int)pad[0] + (int)pad[1] - filters[0][0].rows)/(int)stride[0] + 1, (input[0].cols + (int)pad[2]+(int)pad[3] - filters[0][0].cols)/(int)stride[1] + 1, CV_64FC1);
		  
		  for(int j = in_g*(g-1); j < in_g*g; j++)
		  {
		    Mat temp;
		    
		    temp = CNN_conv(input[j], filters[i][filter_index], stride, pad);
			
 
		    temp1 =  temp1 + temp;
		    filter_index++; 

		 }
		   add(temp1, baises[i], temp1);
		   ret.push_back(temp1);
	  }
  }

  /*if(strcmp(type, "relu")==0)
  {
    for(uint i = 0; i < filters.size(); i++)
	max(ret[i], 0, ret[i]);
  }*/
  return ret;
}

Mat CNN_conv(Mat input, Mat filter, vector<double> stride, vector<double> pad)
{
	Mat ret = Mat::zeros((input.rows + (int)pad[0]+(int)pad[1] - filter.rows)/(int)stride[0] + 1, (input.cols + (int)pad[2]+(int)pad[3] - filter.cols)/(int)stride[1] + 1, CV_64FC1);

	if(pad[0] > 0.0)
    	 copyMakeBorder(input,input,pad[0],pad[1],pad[2],pad[3],BORDER_CONSTANT,Scalar(0));
    	 

	for(int r = 0; r < ret.rows; r++)
       {
    	for(int c = 0; c < ret.cols; c++)
    	{

    		int y_start = stride[1] * r;
    		int x_start = stride[0] * c;
    		Mat patch = input(Range(y_start,y_start+filter.rows), Range(x_start,x_start+filter.cols));
    		Mat temp;
    		multiply(patch,filter,temp);
    		Scalar val = sum(temp);
    		ret.at<double>(r,c) =  val[0];    		
    	}
    }
    return ret;

}
