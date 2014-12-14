/*
 * pooling_layer.cpp
 *
 *  Created on: Nov 24, 2014
 *      Author: manoj
 */

#include <opencv/highgui.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <pooling_layer_bp.h>
using namespace std;
using namespace cv;

vector<Mat> max_pooler_bp(vector<Mat> input, vector<Mat> dzdy, vector<double> pad,  vector<double> stride, vector<double> pool)
{

 int dzdy_h = dzdy[0].rows;
 int dzdy_w = dzdy[0].cols;
 int r,c;
 vector<Mat> dzdx;

  int dzdx_h = input[0].rows;
  int dzdx_w = input[0].cols;
  int outputmaps = input.size();
 
  for(int out = 0; out < outputmaps; out++)
  {
    Mat max_mat = Mat::zeros(dzdx_h, dzdx_w, CV_64FC1);
   for(int dzdy_y = 0; dzdy_y < dzdy_h; dzdy_y++)
   {
	  for(int dzdy_x = 0; dzdy_x < dzdy_w; dzdy_x++)
	  {
		  int y1 = 0, y2 = dzdx_h, x1 = 0, x2 = dzdx_w;
		  if((dzdy_y * (int)stride[0] - (int)pad[0]) > 0)
		     y1 = dzdy_y * (int)stride[0] - (int)pad[0];

		  if( (dzdy_x * (int)stride[1] - (int)pad[1]) > 0)
		    x1 = dzdy_x * (int)stride[1] - (int)pad[1];

		  if( (y1+(int)pool[0]) < dzdx_h)
		    y2 =  y1+(int)pool[0];
	
		  if( (x1+pool[1]) < dzdx_w)
		    x2 = x1+pool[1];

		  double maxVal = -2.2204e-16;//eps
		  for(int dzdx_y = y1; dzdx_y < y2; dzdx_y++)
 		    for(int dzdx_x = x1; dzdx_x < x2 ; dzdx_x++)
			if(input[out].at<double>(dzdx_y, dzdx_x) > maxVal) 
			{
			  maxVal = input[out].at<double>(dzdx_y, dzdx_x);
			  r = dzdx_y;
			  c = dzdx_x;				  
			}
		  max_mat.at<double>(r,c) = max_mat.at<double>(r,c) + dzdy[out].at<double>(dzdy_y, dzdy_x);
	}
   }
   dzdx.push_back(max_mat);   

 }
  
return dzdx;


}

