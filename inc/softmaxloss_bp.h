#ifndef SOFTMAX_LOSS_BP_H_
#define SOFTMAX_LOSS_BP_H_

#include<opencv/cv.h>
#include<vector>
vector<Mat> softmaxloss_bp(vector<Mat> P, int type, float dzdy);

#endif
