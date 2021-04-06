#ifndef __BOXFILTER_H__
#define __BOXFILTER_H__

void naive_boxFilter(float* src, float* dst, int width, int height, int radius);
void boxFilterOpenCVTrick(float* src, float* dst, int width, int height, int radius, float* cache);
void print_res(unsigned char* res, int width,int height);
void print_res(float* res, int width,int height);
void boxFilterOpencvTrick2(float* src, float* dst, int width, int height, int radius, float* cache);
void boxFilterOpenCVTrick3(float* src, float* dst, int width, int height, int radius, float* cache);
void boxFilterArmNeon(float* src, float* dst, int width, int height, int radius, float* cache);
void boxFilterArmNeon_assem(float* src, float* dst, int width, int height, int radius, float* cache);
void absval_naive(int* blob, int w, int h);
void absval_arm_neon(int* blob, int w, int h);
void absval_neon_assem(int* blob, int w, int h);
void boxFilter_convolutiondepthwise_naive(float* src, float* dst, int width, int height, int radius);
void boxFilterConvolutionDepthwise_neon(float* src, float* dst, int width, int height, int radius);
void boxFilterConvolutionDepthwise_assem(float* src, float* dst, int width, int height, int radius);

void BoxFilterBetterOrigin(float *Src, float *Dest, int Width, int Height, int Radius);
void BoxFilterBetterNeonIntrinsics(float *Src, float *Dest, int Width, int Height, int Radius);
#endif