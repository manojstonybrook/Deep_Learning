/*
 * convnet.h
 *
 *  Created on: Oct 4, 2014
 *      Author: manoj
 */

#ifndef CONVNET_H_
#define CONVNET_H_

#include <opencv/cv.h>
#include <string>
#include <vector>
using namespace cv;

//We can include this later in another file
struct featuremap{
	   int id;
	   vector<double> bais;
	   vector<  vector<Mat>  > features;
	   int feature_size;
   };

struct layers{
	    vector<Mat> a; // image transformation value at each layer
	    vector<Mat> d; // For storing delta at each layer
	    vector< vector<Mat> > dk;
	    vector< double > db;
   };


class Convnet
{

  public:
	Convnet();
	Convnet(int neuron_size, int layers, int filters, int output);
	virtual ~Convnet();
	//void cnnfp(Mat training_data);
	void features_init();
	int neuronsz;    //  Number of neurons active   (5X5 in MNSIT)
	int n_filters;   //  Number of convolutional filters layers (16 in MNSIT)
	int layersz;     //  Number of convolutional layers (2 convolutional 2 pooling)
	int n_output;
	int fvnum;
    featuremap featuremap1, featuremap2;
    Mat ffw;
    Mat ffb;
    layers layer1, layer2, layer3, layer4;
    Mat output;
    Mat e;
    Mat od;   //Output Delta
    Mat fvd; // Feature Vector Delta
    Mat dffW;
    Mat dffb;
    Mat fv;
    double L;
  private:
    void feature_init();

};

#endif /* CONVNET_H_ */
