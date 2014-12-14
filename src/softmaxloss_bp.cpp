#include <opencv/highgui.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <softmaxloss.h>
#include<opencv/cv.h>
using namespace std;
using namespace cv;

vector<Mat> softmaxloss_bp(vector<Mat> prob, int type, float dzdy)
{

 double maxVal= prob[0].at<double>(0,0);
 for(int i = 1; i < prob.size(); i++)
 {
    if(prob[i].at<double>(0,0) > maxVal)
      maxVal = prob[i].at<double>(0,0);
 }

 double sum = 0;
 for(int i = 0; i < prob.size(); i++)
 {
   prob[i].at<double>(0,0) -= maxVal;
   sum += exp(prob[i].at<double>(0,0));
 }
 
 for(int i = 0; i < prob.size(); i++)
 {
   prob[i].at<double>(0,0) = exp(prob[i].at<double>(0,0))/sum;
 }

 prob[type].at<double>(0,0) = prob[type].at<double>(0,0) - 1;
 return prob;  
 
}

