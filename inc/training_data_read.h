/*
 * training_data_read.h
 *
 *  Created on: Oct 4, 2014
 *      Author: manoj
 */

#ifndef TRAINING_DATA_READ_H_
#define TRAINING_DATA_READ_H_
#include <opencv/highgui.h>
#include <fstream>
#include <convolution_layer.h>
#include <pooling_layer.h>
#include <norm_layer.h>

void read_norm_layer(char * input, norm_layers  *layer);
void read_pool_layer(char * input, pool_layers  *layer);
void read_conv_layer(char * input, conv_layers *data);
void read_normalize(char * input, Mat data, int width, int height, int nchannels);

#endif /* TRAINING_DATA_READ_H_ */
