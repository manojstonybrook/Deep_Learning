#include <opencv/highgui.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <norm_layer_bp.h>
using namespace std;
using namespace cv;

vector<Mat> norm_bp(vector<Mat> input, vector<double> param, vector<Mat> dzdy)
{

	vector<Mat> dzdx;
	vector<Mat> L;
	
	int y_h = input[0].rows;
	int y_w = input[0].cols;
	int outputmaps = input.size();
	int depth = param[0];
	double k = param[1];
	double alpha = param[2];
	double beta = param[3];
	int tap = depth - (depth-1)/2;

	for(int out = 0; out < outputmaps; out++)
        {		
	  Mat norm(y_h, y_w, CV_64FC1);		
	  for(int y_y = 0; y_y < y_h; y_y++)
	  {
		for(int y_x = 0; y_x < y_w; y_x++)
		{
			double s = 0;
			for(int d = 0; d < depth; d++)
			{
				int idx = out - tap + d + 1;
				if(idx >= 0 && idx < outputmaps)
					s = s + pow(input[idx].at<double>(y_y,y_x), 2);
			}
			norm.at<double>(y_y,y_x) = (k + alpha*s);
		}
	  }
	  
	  L.push_back(norm);
	}


	for(int out = 0; out < outputmaps; out++)
        {		
	  Mat norm(y_h, y_w, CV_64FC1);		
	  for(int y_y = 0; y_y < y_h; y_y++)
	  {
		for(int y_x = 0; y_x < y_w; y_x++)
		{
			double s = 0;
			for(int d = 0; d < depth; d++) 
			{
				int idx = out - tap + d + 1;
				if(idx >= 0 && idx < outputmaps)
					s = s + (dzdy[idx].at<double>(y_y,y_x)/(pow(L[idx].at<double>(y_y,y_x), (beta+1)))*input[idx].at<double>(y_y,y_x)*input[out].at<double>(y_y,y_x));
			}
			norm.at<double>(y_y,y_x) = dzdy[out].at<double>(y_y,y_x)/ (pow(L[out].at<double>(y_y,y_x),beta))-2*alpha*beta*s;
		}
	  }
	  
	  dzdx.push_back(norm);
	}




	return dzdx;

}



