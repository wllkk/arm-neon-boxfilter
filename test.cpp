#include <boxFilter.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <stdlib.h>

using namespace cv;
using namespace std;
void absval_test01();

int main(int argc, char** argv)
{
    Mat src = imread("bus.jpg",0);

    if(src.empty())
    {
        cout << "read pic failed" << endl;
        return -1;
    }
    else
        cout << "read pic successful" << endl;

    int Height = src.rows;
    int Width = src.cols;
    int Radius = 3;
    
    int OutHeight = Height - Radius + 1;
    int OutWidth = Width - Radius + 1;

    unsigned char* Src = src.data;
    
    float* Dest2 = new float[OutHeight * OutWidth];
    float* Dest = new float[Height * Width];
    float* Src_float = new float[Height * Width];
    //float* Dest2 = new float[Height * Width];
    float* cache = new float[Height * Width];

    for(int i = 0; i < Height * Width; i++)
    {
        Src_float[i] = (float)(Src[i]);
    }

    int64 st = cvGetTickCount();

    for(int i = 0; i < 100;i++)
    {
        //naive_boxFilter(Src_float, Dest, Width,Height,Radius);
        //boxFilterOpenCVTrick(Src_float,Dest,Width,Height,Radius,cache);
        //boxFilterOpencvTrick2(Src_float, Dest, Width, Height, Radius, cache);
        //boxFilterOpenCVTrick3(Src_float, Dest, Width, Height, Radius, cache);
        //boxFilterArmNeon(Src_float, Dest, Width, Height, Radius, cache);
        //boxFilterArmNeon_assem(Src_float, Dest, Width, Height, Radius, cache);
        
        //深度可分离卷积方法
        boxFilter_convolutiondepthwise_naive(Src_float, Dest2, Width, Height, Radius);
        //boxFilterConvolutionDepthwise_neon(Src_float, Dest2, Width, Height, Radius);
        //boxFilterConvolutionDepthwise_assem(Src_float, Dest2, Width, Height, Radius);

        //BoxFilterBetterOrigin(Src_float, Dest2, Width, Height, Radius);
        //BoxFilterBetterNeonIntrinsics(Src_float, Dest2, Width, Height, Radius);
    }

    double duration = (cv::getTickCount() - st) / cv::getTickFrequency() * 100;
	//print_res(Dest,Width,Height);
    print_res(Dest2,OutWidth,OutHeight);
    //print_res(Dest2,Width,Height);
    printf("%.5f\n", duration);

    //absval_test01();

    return 0;
}


void absval_test01()
{
    int* blob = new int[100];
    srand((unsigned int)time(0));
    for(int i = 0; i < 100; i++)
    {
        blob[i] = rand() % 100 - 50;
        printf("%d ", blob[i]);
    }
    printf("\n");
    //absval_naive(blob, 20, 5);
    //absval_arm_neon(blob, 20, 5);
    absval_neon_assem(blob, 20, 5);
    for(int i = 0; i < 100; i++)
    {
        printf("%d ", blob[i]);
    }
    printf("\n");
}