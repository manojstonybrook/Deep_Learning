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

int reverseInt (int i);
void read_information(char* file, int &number_of_images, int &n_rows, int &n_cols);
void read_data_cpp(char * input, char *output,  Mat data, Mat label, int batchsize, int start);
void read_data_cpp_match(char * input, char *output, Mat data, Mat label, int batchsize, int start);

#endif /* TRAINING_DATA_READ_H_ */
