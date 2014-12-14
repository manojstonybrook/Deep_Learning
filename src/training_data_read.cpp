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
using namespace std;
using namespace cv;

int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_data_cpp_match(char * input, char *output, Mat data, Mat label, int batchsize, int start)
{

	    // Represent MNIST datafiles as C++ file streams f1 and f2 respectively
		ifstream f1(input,ios::in | ios::binary); // image data
		ifstream f2(output,ios::in | ios::binary); // label data

		if (!f1.is_open() || !f2.is_open())
		{
			cerr << "ERROR: Can't open MNIST files. Please locate them in current directory" << endl;
			return;
		}
		// Create buffers for image data and correct labels
		char *header = new char[2];
		int n_rows = 28;
		int n_cols = 28;
		//f1.read(buffer,16);
		//f2.read(buffer,8);

		f1.seekg ( start*n_rows*n_cols ); // Header + number of images shift
		f2.seekg ( 10*start);

		for(int n = 0; n < batchsize; n++)
		{
			for(int r=0;r < n_rows; ++r)
		    {
			  for(int c=0; c < n_cols; ++c)
			  {
				unsigned char temp = 0;
				f1.read((char*)&temp,sizeof(temp));
				data.at<Vec<double, 50> >(r,c)(n)=(double)temp/255;
			  }
		    }
			unsigned char temp=0;
			for(int k = 0; k < 10; k++)
			{
		 	 f2.read((char*)&temp,sizeof(temp));
			 label.at<double>(k,n) = (double)temp;
			}
		}


}




void read_data_cpp(char * input, char *output, Mat data, Mat label, int batchsize, int start)
{

	    // Represent MNIST datafiles as C++ file streams f1 and f2 respectively
		ifstream f1(input,ios::in | ios::binary); // image data
		ifstream f2(output,ios::in | ios::binary); // label data

		if (!f1.is_open() || !f2.is_open())
		{
			cerr << "ERROR: Can't open MNIST files. Please locate them in current directory" << endl;
			return;
		}
		// Create buffers for image data and correct labels
		const int BUF_SIZE = 2048;
		char *buffer = new char[BUF_SIZE];
		char *header = new char[2];
		int n_rows = 28;
		int n_cols = 28;
		//f1.read(buffer,16);
		//f2.read(buffer,8);

		f1.seekg ( 16 + start*n_rows*n_cols ); // Header + number of images shift
		f2.seekg ( 8 + start);

		for(int n = 0; n < batchsize; n++)
		{
			for(int r=0;r < n_rows; ++r)
		    {
			  for(int c=0; c < n_cols; ++c)
			  {
				unsigned char temp = 0;
				f1.read((char*)&temp,sizeof(temp));
				data.at<Vec<double, 50> >(r,c)(n)=(double)temp/255;
			  }
		    }
			unsigned char temp=0;
			f2.read((char*)&temp,sizeof(temp));
			int pos = (int) temp;
			label.at<double>(pos,n) = 1;
		}


}

void read_information(char* name, int &number_of_images, int &n_rows, int &n_cols)
{
    ifstream file(name);

    if (file.is_open())
    {
        int magic_number=0;
        number_of_images=0;
        n_rows=0;
        n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        cout << "\n Number of images to train:" << number_of_images << "\n Rows" << n_rows << "\n Cols:" << n_cols << "\n";

    }
    else
    	cout << "Not able to open this file";

}


