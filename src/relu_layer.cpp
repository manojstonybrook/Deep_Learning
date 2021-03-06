/*
 * relu_layer.cpp
 *
 *  Created on: Nov 23, 2014
 *      Author: manoj
 */

#include <opencv/highgui.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <relu_layer.h>
using namespace std;
using namespace cv;

vector<Mat> relu(vector<Mat> input)
{

for(uint i = 0; i < input.size(); i++)
   max(input[i], 0, input[i]);

return input;


}


vector<Mat> relu_bp(vector<Mat> dz, vector<Mat> input)
{

for(uint i = 0; i < input.size(); i++)
  {
    for(int r = 0; r < input[0].rows; r++)
     for(int c = 0; c < input[0].cols; c++)
        if(input[i].at<double>(r,c) <= 0)
           dz[i].at<double>(r,c) = 0;
  }
return dz;

}
