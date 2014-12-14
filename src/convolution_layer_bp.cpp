#include <opencv/highgui.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sigmoid.h>
#include <convolution_layer_bp.h>
using namespace std;
using namespace cv;

vector<Mat> CNN_BP(vector<Mat> input,  vector< vector <Mat> > filters, vector<Mat> dzdy, vector<float> baises, vector<double> stride, vector<double> pad)
{

//fully connected mode
int fc = 0;
if(filters[0][0].rows == input[0].rows && filters[0][0].cols == input[0].cols && filters[0].size() == input.size())
  fc = 1;

vector<Mat> dzdx, dzdx1;
// outputmap height and width calculation
//int y_h = ((input[0].rows - filters[0][0].rows + pad[1] + pad[2])/stride[1]) + 1;
//int y_w = (input[0].cols - filters[0][0].rows + pad[3] + pad[4])/stride[2]) + 1;
int y_h = dzdy[0].rows;
int y_w = dzdy[0].cols;
// filters height and width
int f_h = filters[0][0].rows;
int f_w = filters[0][0].cols;

//this is for the multiple machine execution, 
// no information exchange between these groups
int group = input.size()/filters[0].size();
// the number of inputmaps
int in_g = filters[0].size();
//the number of outputmaps for each group
int out_g = filters.size()/group;

vector<Mat> input1 = input;
vector<float> dzdb(baises);

for(int i = 0; i < input.size(); i++)
  copyMakeBorder(input[0],input1[0],pad[0],pad[1],pad[2],pad[3],BORDER_CONSTANT,Scalar(0));

/*
vector< vector<Mat> > dzdw;

  for(int g = 1; g <= group; g++)
  {
	  for(int i = out_g*(g-1); i < out_g*g; i++ )
	  {
		  vector<Mat> temp1;// = Mat::zeros(filters[0][0].rows,filters[0][0].cols, CV_64FC(filters.size()));

		  for(int j = in_g*(g-1); j < in_g*g; j++)
		  {
		    Mat temp = Mat::zeros(filters[0][0].rows,filters[0][0].cols, CV_64FC1);
		    for(int y_y = 0; y_y < y_h; y_y++)
		    {
		      for(int y_x = 0; y_x < y_w; y_x++)
		      {
			   for(int f_y = 0; f_y < f_h; f_y++)
			   {
			     for(int f_x = 0; f_x < f_w; f_x++)
			     {
								
				int x1 = y_y * stride[0] + f_y;
				int x2 = y_x * stride[1] + f_x;
				temp.at<double>(f_y,f_x) = temp.at<double>(f_y,f_x) + dzdy[i].at<double>(y_y,y_x)*input1[j].at<double>(x1,x2);

			     }

			  }
			 
		      }

		   }
		  temp1.push_back(temp);

	       }
		       
	     dzdw.push_back(temp1);
         }		 
 }


for(int out = 0; out < out_g*group; out++)
{
 dzdb[out] = 0;
 for(int y_y = 0; y_y < y_h; y_y++)
   for(int y_x = 0; y_x < y_w; y_x++)
	dzdb[out] = dzdb[out] + dzdy[out].at<double>(y_y,y_x);
}
*/
//for(int i = 0; i < out_g*group; i++)
 //cout << "   " << i << "   "<<dzdb[i] << " " << dzdy[i].at<double>(0,0);


int x_h = input[0].rows;
int x_w = input[0].cols;
vector<Mat> filter1;

int sx = x_h;
int sy = x_w;
int sc = input.size();
if(fc == 1)
{
// x and filter are resphaped
x_h = 1;
x_w = 1;
f_h = 1;
f_w = 1; 
in_g =  input.size() * input[0].rows * input[0].cols;

for(int i = 0; i < filters.size(); i++)
{ 
 Mat temp;
 for(int ch = 0; ch < filters[0].size(); ch++)
   {
    for(int c = 0; c < filters[0][0].cols; c++)
     for(int r = 0; r < filters[0][0].rows; r++)
        temp.push_back(filters[i][ch].at<double>(r,c));

   }
 filter1.push_back(temp);
}

}
vector<Mat> padded_dzdy;
for(int i = 0; i < out_g*group; i++)
{
  Mat temp = Mat::zeros(x_h+f_h-1, x_w+f_w-1, CV_64FC1);
  padded_dzdy.push_back(temp);
}

int t = (padded_dzdy[0].rows - x_h)/2;

if(stride[0]>1)
{
  for(int out=0; out < out_g*group; out++)
   for(int y_y = 0; y_y < y_h; y_y++)
    for(int y_x = 0; y_x < y_w; y_x++)
	padded_dzdy[out].at<double>(t+(y_y)*stride[0]+1,t+(y_x)*stride[1]+1) = dzdy[out].at<double>(y_y,y_x);
  
}
else
{
  for(int out=0; out < out_g*group; out++)
   for(int y_y = 0; y_y < y_h; y_y++)
    for(int y_x = 0; y_x < y_w; y_x++)
       {
	padded_dzdy[out].at<double>(t+(y_y),t+(y_x)) = dzdy[out].at<double>(y_y,y_x);
	//cout << "in" << padded_dzdy[out].at<double>(t+(y_y),t+(y_x));
       }
}

//compute dzdx
for(int g = 1; g <= group; g++)
{
 for(int in = 0; in < in_g; in++)
 {
  Mat temp = Mat::zeros(x_h, x_w, CV_64FC1);
  for(int out = 0; out < out_g; out++)
  {
    int in_idx = in+(g-1)*in_g;
    int out_idx = out+(g-1)*out_g;
   for(int x_y = 0; x_y < x_h; x_y++)
     for(int x_x = 0; x_x < x_w; x_x++)
       for(int f_y = 0; f_y < f_h; f_y++)
        for(int f_x = 0; f_x < f_w; f_x++)
         {

	   int y1 = x_y+f_y; 
	   int y2 = x_x+f_x;
	   if(fc == 1)
	   temp.at<double>(x_y,x_x) = temp.at<double>(x_y,x_x) + padded_dzdy[out_idx].at<double>(y1,y2) * filter1[out_idx].at<double>(0, (f_h-f_y-1)*f_w + (f_w-f_x-1)+in_idx);	
	   else
	   temp.at<double>(x_y,x_x) = temp.at<double>(x_y,x_x) + padded_dzdy[out_idx].at<double>(y1,y2)*filters[out_idx][in].at<double>(f_h-f_y-1,f_w-f_x-1);	
	 }
  }
    	  
     dzdx1.push_back(temp);
 }
}


if(fc == 1)
{
  
  int count= 0;
  for(int t = 0; t < dzdx1.size()/(sx*sy); t++)
  {
   Mat temp1 = Mat::zeros(sy, sx,CV_64FC1);
   for(int c = 0; c < sx; c++)
     for(int r = 0; r < sy; r++)
      {
       temp1.at<double>(r,c) = dzdx1[count].at<double>(0,0);
       count++;
      }
    dzdx.push_back(temp1); 
  }
 
 return dzdx;   

}
else
return dzdx1; 
 
//for(int i = 0; i < 4096; i++)
 //cout << " "<< dzdx[i];

//cout << "\n size" << dzdx.size() << " " << dzdx[0].rows << " " << dzdx[0].cols;
//cout << t << " " << padded_dzdy.size() << " "<<padded_dzdy[0].rows << " "<<padded_dzdy[1].cols;
  
}

