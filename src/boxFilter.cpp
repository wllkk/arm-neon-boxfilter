#include <iostream>
#include <opencv2/opencv.hpp>
#include <boxFilter.hpp>
#include <arm_neon.h>


using namespace cv;

typedef unsigned char uchar;


void print_res(unsigned char* res, int width,int height)
{
    for(int h = 0; h < height - 50; h+=50)
    {
        for(int w = 0; w < width - 50; w+=50)
        {
            printf("%d ",res[h * width + w]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_res(float* res, int width,int height)
{
    for(int h = 0; h < height - 50; h+=50)
    {
        for(int w = 0; w < width - 50; w+=50)
        {
            printf("%.1f ",res[h * width + w]);
        }
        printf("\n");
    }
    printf("\n");
}

void naive_boxFilter(float* src, float* dst, int width, int height, int radius)
{
    for(int h = 0; h < height;h++)
    {
        int height_sift = h * width;
        for(int w = 0; w < width; w++)
        {
            int start_h = h - radius;
            if(start_h < 0) start_h = 0;
            int end_h = h + radius;
            if(end_h > height - 1) end_h = height -1;
            int start_w = w - radius;
            if(start_w < 0) start_w = 0;
            int end_w = w + radius;
            if(start_w > width - 1) start_w = width - 1;

            float temp = 0;
            for(int sh = start_h; sh <= end_h; sh++)
            {
                for(int sw = start_w; sw <= end_w; sw++)
                {
                    temp += src[sh * width + sw];
                }
            }
            dst[height_sift + w] = temp;
        }
    }
}

//行 列 分离运算
void boxFilterOpenCVTrick(float* src, float* dst, int width, int height, int radius, float* cache)
{
    //处理行
    for(int h = 0; h < height; h++)
    {
        int height_sift = h * width;
        for(int w = 0; w < width; w++)
        {
            int start_w = w - radius;
            if(start_w < 0) start_w = 0;

            int end_w = w + radius;
            if(end_w > height - 1) end_w = height -1;

            float temp = 0;
            for(int sw = start_w; sw <= end_w; sw++)
            {
                temp += src[height_sift + sw];
            }
            cache[height_sift + w] = temp;
        }
    }

    //处理列
    for(int h = 0; h < height; h++)
    {
        int height_sift = h * width;

        int start_h = h - radius;
        if(start_h < 0) start_h = 0;

        int end_h = h + radius;
        if(end_h > height -1) end_h = height - 1;

        //最外层的循环次数多
        // for(int w = 0; w < width; w++)
        // {
        //     float temp = 0;
        //     for(int sh = start_h; sh <= end_h; sh++)
        //     {
        //         temp += cache[sh * width + w];
        //     }
        //     dst[height_sift + w] = temp;
        // }

        //最外层的循环次数少
        for(int sh = start_h; sh <= end_h; ++sh)
        {
            for(int w = 0; w < width; ++w)
            {
                dst[sh * width + w] += cache[height_sift + w];
            }
        }

    }
}

//行列展开 滑窗计算  维护 temp 变量
void boxFilterOpencvTrick2(float* src, float* dst, int width, int height, int radius, float* cache)
{
    for(int h = 0; h < height; ++h)
    {
        int height_sift = h * width;

        // head
        float temp = 0;
        for(int w = 0; w < radius; w++)
        {
            temp += src[height_sift + w];
        }

        for(int w = 0; w <= radius; w++)
        {
            temp += src[height_sift + w + radius];
            cache[height_sift + w] = temp;
        }

        //middle
        int start = radius + 1;
        int end = width - 1 - radius;
        for(int w = start; w <=end; ++w)
        {
            temp += src[height_sift + w + radius];
            temp -= src[height_sift + w - radius - 1];
            cache[height_sift + w] = temp;
        }

        //tail
        for(int w = width -radius; w < width;w++)
        {
            temp -= src[height_sift + w - radius - 1];
            cache[height_sift + w] = temp;
        }
    }

    //列操作
    for(int w = 0; w < width; ++w)
    {
        float temp = 0;
        //head
        for(int h = 0; h < radius; ++h)
        {
            temp += cache[h * width + w];
        }

        for(int h = 0; h <= radius; ++h)
        {
            temp += cache[(h + radius) * width + w]; //列操作
            dst[h * width + w] = temp;
        }

        //middle
        int start = radius + 1;
        int end = height - 1 - radius;
        
        for(int h = start; h <= end; ++h)
        {
            temp += cache[(h + radius) * width + w];
            temp -= cache[(h - radius - 1) * width + w]; //列跨行操作 cache miss
            dst[h * width + w] = temp;
        }

        //tail

        for(int h = height - radius; h < height; ++h)
        {   
            temp -= cache[(h - radius - 1) * width + w];
            dst[h * width + w] = temp;
        }
    }
}

// 列操作优化 ，减少不必要的 CPU Cache miss
void boxFilterOpenCVTrick3(float* src, float* dst, int width, int height, int radius, float* cache)
{
    //处理行 ， 和上面的一样，不用改
    for(int h = 0; h < height; ++h)
    {
        float temp = 0;
        int height_sift = h * width;
        //head
        for(int w = 0; w < radius; ++w)
        {
            temp += src[height_sift + w];
        }

        for(int w = 0; w <= radius; ++w)
        {
            temp += src[height_sift + radius + w];
            cache[height_sift + w] = temp;
        }

        //middle
        int start = radius + 1;
        int end = width - 1 - radius;
        for(int w = start; w <= end; w++)
        {
            temp += src[height_sift + radius + w];
            temp -= src[height_sift + w - radius - 1];
            cache[height_sift + w] = temp;
        }

        //tail
        for(int w = width - radius; w < width; w++)
        {
            temp -= src[height_sift + w - radius - 1];
            cache[height_sift + w] = temp;
        }
    }

    //列处理
    std::vector<float> colsum(width, 0);
    float* ptrcolsum = &(colsum[0]);
   
    //把三行数据拿过来 加在一行上，后面操作维护这个 colsum  ，压一行  弹一行 
    for(int x = 0; x < radius; ++x)
    {
        int sift = x * width;
        for(int y = 0; y < width;y++)
        {
            ptrcolsum[y] += cache[sift + y];  //?
        }        
    }    
    // for(auto it = colsum.begin(); it != colsum.end();it++)
    // {
    //     std::cout << *it << " ";
    // }
    // std::cout << std::endl;
    //head
    for(int h = 0; h <= radius; h++)
    {
        for(int w = 0; w < width; ++w)
        {
            ptrcolsum[w] += cache[(h + radius) * width + w];
            dst[h * width + w] = ptrcolsum[w];
        }
    }
   //printf("arm:\n");
    //print_res(dst, width,height);    
    //middle
    int start = radius + 1;
    int end = height - radius -1;
    for(int h = start; h <= end; ++h)
    {
        int sift = h * width;
        for(int w = 0; w < width; ++w)
        {
            ptrcolsum[w] += cache[(h + radius) * width + w];
            ptrcolsum[w] -= cache[(h - radius - 1) * width + w];
            dst[sift + w] = ptrcolsum[w];
        }
    }

    //tail
    for(int h = height - radius; h < height; h++)
    {
        int sift = h * width;
        for(int w = 0; w < width; w++)
        {
            ptrcolsum[w] -= cache[(h - radius - 1) * width + w];
            dst[sift + w] = ptrcolsum[w];
        }
    }

}

void boxFilterArmNeon(float* src, float* dst, int width, int height, int radius, float* cache)
{
    //处理行 ， 和上面的一样，不用改
    for(int h = 0; h < height; ++h)
    {
        float temp = 0;
        int height_sift = h * width;
        //head
        for(int w = 0; w < radius; ++w)
        {
            temp += src[height_sift + w];
        }

        for(int w = 0; w <= radius; ++w)
        {
            temp += src[height_sift + radius + w];
            cache[height_sift + w] = temp;
        }

        //middle
        int start = radius + 1;
        int end = width - 1 - radius;
        for(int w = start; w <= end; w++)
        {
            temp += src[height_sift + radius + w];
            temp -= src[height_sift + w - radius - 1];
            cache[height_sift + w] = temp;
        }

        //tail
        for(int w = width - radius; w < width; w++)
        {
            temp -= src[height_sift + w - radius - 1];
            cache[height_sift + w] = temp;
        }
    }

    //列处理
    std::vector<float> colsum(width, 0);
    float* ptrcolsum = &(colsum[0]);
    
//    //列优化填充初始数据
//     for(int h = 0; h < radius; ++h)
//     {
//         for(int w = 0; w < width; w++)
//         {
//             ptrcolsum[w] += cache[h * width + w];
//         }
//     }

#if __ARM_NEON 
    int block = width >> 2;  // 4 的倍数次读取
    int remain = width - (block << 2); //  余数
    for(int h = 0; h < radius; ++h)
    {
        float* tmpcolsum = ptrcolsum;
        float* tmpcache = cache + h * width;

        int n = block;
        int r = remain;

        for( ;n > 0;--n) 
        {
            float32x4_t coldata = vld1q_f32(tmpcolsum); // 内存 -> 寄存器
            float32x4_t cachedata = vld1q_f32(tmpcache); //内存 -> 寄存器

            float32x4_t sum = vaddq_f32(coldata, cachedata); // 四个字节对应相加

            vst1q_f32(tmpcolsum, sum); //寄存器 -> 内存

            tmpcolsum += 4; //地址偏移
            tmpcache += 4;
        }

        for(; r > 0; r--)
        {
            *tmpcolsum += * tmpcache;
            tmpcolsum++;
            tmpcache++;
        }
    }
#else
    //列优化填充初始数据
    for(int h = 0; h < radius; ++h)
    {
        for(int w = 0; w < width; ++w)
        {
            ptrcolsum[w] += cache[h * width + w];
        }
    }
#endif
    // for(auto it = colsum.begin(); it != colsum.end();it++)
    // {
    //     std::cout << *it << " ";
    // }
    // std::cout << std::endl;
//head
#if __ARM_NEON
    for(int h = 0; h <= radius; h++)
    {
        float* tmpcache = cache + (h + radius) * width;
        float* tmpcolsum = ptrcolsum;
        float* tmpdst = dst + h * width;

        int n = block;
        int r = remain;

        for(; n > 0; n--)
        {
            float32x4_t add = vld1q_f32(tmpcache);
            float32x4_t coldata = vld1q_f32(tmpcolsum);
            coldata = vaddq_f32(add, coldata);
            
            vst1q_f32(tmpcolsum, coldata);
            vst1q_f32(tmpdst, coldata);

            tmpcolsum += 4;
            tmpcache += 4;
            tmpdst += 4;
        }
        //处理尾巴
        for(;r > 0; r--)
        {
            *tmpcolsum += *tmpcache;
            *tmpdst += *tmpcolsum;

            tmpdst++;
            tmpcache++;
            tmpcolsum++;
        }
    }

#else
    //head
    for(int h = 0; h <= radius; h++)
    {
        for(int w = 0; w < width; ++w)
        {
            ptrcolsum[w] += cache[(h + radius) * width + w];
            dst[h * width + w] = ptrcolsum[w];
        }
    }
#endif
    //printf("neon:\n");
   // print_res(dst, width,height); 
//middle
#if __ARM_NEON
    int start = radius + 1;
    int end = height - radius - 1;

    for(int h = start; h <= end; ++h)
    {
        int n = block;
        int r = remain;

        float* tmpcacheadd = cache + (h + radius) * width;
        float* tmpcachesub = cache + (h - radius -1) * width;
        float* tmpcolsum = ptrcolsum;
        float* tmpdst = dst + h * width;

        for(; n > 0; n--)
        {
            float32x4_t cachedataAdd = vld1q_f32(tmpcacheadd);
            float32x4_t cachedataSub = vld1q_f32(tmpcachesub);
            float32x4_t colsumdata = vld1q_f32(tmpcolsum);
            
            float32x4_t sumdata = vaddq_f32(colsumdata, cachedataAdd);
            sumdata = vsubq_f32(sumdata, cachedataSub);

            vst1q_f32(tmpcolsum, sumdata);
            vst1q_f32(tmpdst, sumdata);

            tmpcacheadd += 4;
            tmpcachesub += 4;
            tmpcolsum += 4;
            tmpdst += 4;
        }

        for(; r > 0; r--)
        {
            *tmpcolsum += *tmpcacheadd;
            *tmpcolsum -= *tmpcachesub;
            *tmpdst = *tmpcolsum;

            tmpcolsum++;
            tmpcacheadd++;
            tmpcachesub++;
            tmpdst++;
        }

    }
#else
    int start = radius + 1;
    int end = height - radius - 1;

    for(int h = start; h <= end; ++h)
    {
        int sift = h * width;
        for(int w = 0; w < width; ++w)
        {
            ptrcolsum[w] += cache[(h + radius) * width + w];
            ptrcolsum[w] -= cache[(h - radius - 1) * width + w];
            dst[sift + w] = ptrcolsum[w];
        }    
    }

#endif

//tail
#if __ARM_NEON

    for(int h = height - radius; h < height; h++)
    {
        int n = block;
        int r = remain;
        
        float* tmpdst = dst + h * width;
        float* tmpcacheSub = cache + (h - radius - 1) * width;
        float* tmpcolsum = ptrcolsum;

        for(; n > 0; n--)
        {
            float32x4_t coldata = vld1q_f32(tmpcolsum);
            float32x4_t cachedata = vld1q_f32(tmpcacheSub);
            float32x4_t sumdata = vsubq_f32(coldata, cachedata);

            vst1q_f32(tmpcolsum, sumdata);
            vst1q_f32(tmpdst, sumdata);

            tmpdst +=4;
            tmpcacheSub += 4;
            tmpcolsum += 4;
        }

        for(;r > 0; r--)
        {
            *tmpcolsum -= *tmpcacheSub;
            *tmpdst = *tmpcolsum;

            tmpcolsum++;
            tmpdst++;
            tmpcacheSub++;
        }
    }

#else
    for(int h = height - radius; h < height; h++)
    {
        for(int w = 0; w < width; w++)
        {
            ptrcolsum[w] -= cache[ (h - radius - 1) * width + w];
            dst[h * width + w] = ptrcolsum[w];
        }
    }
#endif
}

void boxFilterArmNeon_assem(float* src, float* dst, int width, int height, int radius, float* cache)
{
        //处理行 ， 和上面的一样，不用改
    for(int h = 0; h < height; ++h)
    {
        float temp = 0;
        int height_sift = h * width;
        //head
        for(int w = 0; w < radius; ++w)
        {
            temp += src[height_sift + w];
        }

        for(int w = 0; w <= radius; ++w)
        {
            temp += src[height_sift + radius + w];
            cache[height_sift + w] = temp;
        }

        //middle
        int start = radius + 1;
        int end = width - 1 - radius;
        for(int w = start; w <= end; w++)
        {
            temp += src[height_sift + radius + w];
            temp -= src[height_sift + w - radius - 1];
            cache[height_sift + w] = temp;
        }

        //tail
        for(int w = width - radius; w < width; w++)
        {
            temp -= src[height_sift + w - radius - 1];
            cache[height_sift + w] = temp;
        }
    }

    //列处理
    std::vector<float> colsum(width, 0);
    float* ptrcolsum = &(colsum[0]);

#if __ARM_NEON

    int block = width >> 2;  // 4 的倍数次读取
    int remain = width - (block << 2); //  余数
    for(int h = 0; h < radius; ++h)
    {
        float* tmpcolsum = ptrcolsum;
        float* tmpcache = cache + h * width;

        int n = block;
        int r = remain;

        for( ;n > 0;--n) 
        {
            float32x4_t coldata = vld1q_f32(tmpcolsum); // 内存 -> 寄存器
            float32x4_t cachedata = vld1q_f32(tmpcache); //内存 -> 寄存器

            float32x4_t sum = vaddq_f32(coldata, cachedata); // 四个字节对应相加

            vst1q_f32(tmpcolsum, sum); //寄存器 -> 内存

            tmpcolsum += 4; //地址偏移
            tmpcache += 4;
        }

        for(; r > 0; r--)
        {
            *tmpcolsum += * tmpcache;
            tmpcolsum++;
            tmpcache++;
        }
    }
    //head
    for(int h = 0; h <= radius; h++)
    {
        float* tmpcache = cache + (h + radius) * width;
        float* tmpcolsum = ptrcolsum;
        float* tmpdst = dst + h * width;

        int n = block;
        int r = remain;

        for(; n > 0; n--)
        {
            float32x4_t add = vld1q_f32(tmpcache);
            float32x4_t coldata = vld1q_f32(tmpcolsum);
            coldata = vaddq_f32(add, coldata);
            
            vst1q_f32(tmpcolsum, coldata);
            vst1q_f32(tmpdst, coldata);

            tmpcolsum += 4;
            tmpcache += 4;
            tmpdst += 4;
        }
        //处理尾巴
        for(;r > 0; r--)
        {
            *tmpcolsum += *tmpcache;
            *tmpdst += *tmpcolsum;

            tmpdst++;
            tmpcache++;
            tmpcolsum++;
        }
    }

#if __aarch64__

    int start = radius + 1;
    int end = height - radius - 1;

    block = width >> 3;  // 4 的倍数次读取
    remain = width - (block << 3); //  余数

    for(int h = start; h <= end; ++h)
    {
        int n = block;
        int r = remain;
 
        float* tmpcacheadd = cache + (h + radius) * width;
        float* tmpcachesub = cache + (h - radius -1) * width;
        float* tmpcolsum = ptrcolsum;
        float* tmpdst = dst + h * width;

        // __asm__  volatile(
        //     "0:                                 \n"
        //     "prfm      pldl1keep,   [%0, #128]  \n"
        //     "ld1       {v0.4s},     [%0], #16   \n"

        //     "prfm      pldl1keep,    [%1, #128]  \n"
        //     "ld1       {v1.4s},      [%1], #16   \n"
 
        //     "prfm      pldl1keep,   [%2, #128]   \n"
        //     "ld1       {v2.4s},     [%2]         \n"

        //     "fadd       v3.4s, v0.4s, v2.4s      \n"
        //     "fsub       v4.4s, v3.4s, v1.4s      \n"

        //     "st1       {v4.4s},     [%2], #16   \n"
        //     "st1       {v4.4s},     [%3], #16   \n"

        //     "subs      %w4,   %w4,  #1          \n"
        //     "bne       0b                       \n"

        //     : "=r"(tmpcacheadd),
        //     "=r"(tmpcachesub),
        //     "=r"(tmpcolsum),
        //     "=r"(tmpdst),
        //     "=r"(n)
        //     : "0"(tmpcacheadd),
        //     "1"(tmpcachesub),
        //     "2"(tmpcolsum),
        //     "3"(tmpdst),
        //     "4"(n)
        //     : "cc", "memory", "v0","v1","v2","v3","v4","w0"
        // );

        __asm__  volatile(
            "0:                                 \n"
            "prfm      pldl1keep,   [%0, #128]  \n"
            "ld1       {v0.4s, v1.4s},     [%0], #32   \n"

            "prfm      pldl1keep,   [%2, #128]   \n"
            "ld1       {v4.4s, v5.4s},     [%2]         \n"

            "fadd       v6.4s, v0.4s, v4.4s      \n"
            "fadd       v7.4s, v1.4s, v5.4s      \n"

            "prfm      pldl1keep,    [%1, #128]  \n"
            "ld1       {v2.4s, v3.4s},      [%1], #32   \n"

            "fsub       v8.4s, v6.4s, v2.4s      \n"
            "fsub       v9.4s, v7.4s, v3.4s      \n"

            "st1       {v8.4s, v9.4s},     [%2], #32   \n"
            "st1       {v8.4s, v9.4s},     [%3], #32   \n"

            "subs      %w4,   %w4,  #1          \n"
            "bne       0b                       \n"

            : "=r"(tmpcacheadd),
            "=r"(tmpcachesub),
            "=r"(tmpcolsum),
            "=r"(tmpdst),
            "=r"(n)
            : "0"(tmpcacheadd),
            "1"(tmpcachesub),
            "2"(tmpcolsum),
            "3"(tmpdst),
            "4"(n)
            : "cc", "memory", "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","w0"
        );

        for(; r > 0; r--)
        {
            *tmpcolsum += *tmpcacheadd;
            *tmpcolsum -= *tmpcachesub;
            *tmpdst = *tmpcolsum;

            tmpcolsum++;
            tmpcacheadd++;
            tmpcachesub++;
            tmpdst++;
        }

    }
#endif

 for(int h = height - radius; h < height; h++)
    {
        int n = block;
        int r = remain;
        
        float* tmpdst = dst + h * width;
        float* tmpcacheSub = cache + (h - radius - 1) * width;
        float* tmpcolsum = ptrcolsum;

        for(; n > 0; n--)
        {
            float32x4_t coldata = vld1q_f32(tmpcolsum);
            float32x4_t cachedata = vld1q_f32(tmpcacheSub);
            float32x4_t sumdata = vsubq_f32(coldata, cachedata);

            vst1q_f32(tmpcolsum, sumdata);
            vst1q_f32(tmpdst, sumdata);

            tmpdst +=4;
            tmpcacheSub += 4;
            tmpcolsum += 4;
        }

        for(;r > 0; r--)
        {
            *tmpcolsum -= *tmpcacheSub;
            *tmpdst = *tmpcolsum;

            tmpcolsum++;
            tmpdst++;
            tmpcacheSub++;
        }
    }
#endif

}

void absval_naive(int* blob, int w, int h)
{
    int size = w * h;
    for(int i = 0; i < size; i++)
    {
        if(blob[i] < 0)
        {
            blob[i] = -blob[i];
        }
    }
}

void absval_arm_neon(int* blob, int w, int h)
{
    int size = w * h;
    int block = size >> 2;
    int remain = size - (block << 2);

    int* ptrblob = blob;
    for(; block > 0; --block)
    {
        int32x4_t data = vld1q_s32(ptrblob);
        data = vabsq_s32(data);

        vst1q_s32(ptrblob, data);

        ptrblob += 4;
    }

    for(; remain > 0; -- remain)
    {
        if(*ptrblob < 0)
        {
            *ptrblob = - *ptrblob;
        }
        ptrblob++;
    }
}

void absval_neon_assem(int* blob, int w, int h)
{

    int size = w * h;
    int block = size >> 2;
    int remain = size - (block << 2);
    int* ptrblob = blob;

#if __ARM_NEON
#if __aarch64__             //armv8

    if(block > 0)
    {
        asm volatile(
            "0:                                     \n"
            "prfm           pldl1keep, [%1, #128]   \n" //数据预读取 
            "ld1            {v0.4s},   [%1]         \n"
            "abs            v0.4s,   v0.4s          \n"
            "st1            {v0.4s}, [%1], #16      \n"  // v0 寄存器写回ptrblob, ptrblob 地址自增16个字节
            "subs           %w0, %w0, #1            \n"
            "bne            0b                      \n"
            :"=r"(block),
            "=r"(ptrblob)
            :"0"(block),
            "1"(ptrblob)
            :"cc", "memory","v0"
        );
    }
#else                   //armv7
    if(block > 0)
    {
        "0:                                     \n"
        "pld                          [%1, #128]\n"
        : "=r"(block),
        "=r"(ptrblob)
        : "0"(block),
        "1"(ptrblob)
        :"cc", "memory", "q0"
    }

#endif
#endif

    for(; remain > 0; --remain)
    {
        if(*ptrblob < 0)
        {
            *ptrblob = - *ptrblob;
            ptrblob++;
        }
    }
}

void boxFilter_convolutiondepthwise_naive(float* src, float* dst, int width, int height, int radius)
{
    float* kernel = new float[radius * radius];
    for(int i = 0; i < radius * radius; i++)
    {
        kernel[i] = 1.0;
    }

    int OutputWidth = width - radius + 1;
    int OutputHeight = height - radius + 1;

    float* r0 = src;
    float* r1 = src + width;
    float* r2 = r1 + width;
    float* r3 = r2 + width;

    float* k0 = kernel;
    float* k1 = k0 + 3;
    float* k2 = k1 + 3;

    float* outptr = dst;
    float* outptr2 = dst + OutputWidth;

    int i = 0;
    for(; i + 1 < OutputHeight; i += 2)
    {
        int remain = OutputWidth;
        for(; remain > 0; remain --)
        {
            float sum1 = 0, sum2 = 0;

            sum1 += r0[0] * k0[0];
            sum1 += r0[1] * k0[1];
            sum1 += r0[2] * k0[2];
            sum1 += r1[0] * k1[0];
            sum1 += r1[1] * k1[1];
            sum1 += r1[2] * k1[2];
            sum1 += r2[0] * k2[0];
            sum1 += r2[1] * k2[1];
            sum1 += r2[2] * k2[2];

            sum2 += r1[0] * k0[0];
            sum2 += r1[1] * k0[1];
            sum2 += r1[2] * k0[2];
            sum2 += r2[0] * k1[0];
            sum2 += r2[1] * k1[1];
            sum2 += r2[2] * k1[2];
            sum2 += r3[0] * k2[0];
            sum2 += r3[1] * k2[1];
            sum2 += r3[2] * k2[2];
            //printf("%.2f", sum1);
            *outptr = sum1;
            *outptr2 = sum2;
            r0++;
            r1++;
            r2++;
            r3++;
            outptr++;
            outptr2++;
        }
        r0 += 2 + width;
        r1 += 2 + width;
        r2 += 2 + width;
        r3 += 2 + width;

        outptr += OutputWidth;
        outptr2 += OutputWidth;
    }

    for(; i < OutputHeight; i++)
    {
        int remain = OutputWidth;
        for(; remain > 0; --remain)
        {
            float sum = 0.0;
            sum += r0[0] * k0[0];
            sum += r0[1] * k0[1];
            sum += r0[2] * k0[2];

            sum += r1[0] * k1[0];
            sum += r1[1] * k1[1];
            sum += r1[2] * k1[2];

            sum += r2[0] * k2[0];
            sum += r2[1] * k2[1];
            sum += r2[2] * k2[2];

            *outptr = sum;
            r0++;
            r1++;
            r2++;
            outptr++;
        }

        r0 += 2;
        r1 += 2;
        r2 += 2;
    }

}

void boxFilterConvolutionDepthwise_neon(float* src, float* dst, int width, int height, int radius)
{
    int OutputWidth = width - radius + 1;
    int OutputHeight = height - radius + 1;

    float* kernel = new float[radius * radius];
    for(int i = 0; i < radius * radius; i++)
    {
        kernel[i] = 1.0;
    }
    
    float* r0 = src;
    float* r1 = r0 + width;
    float* r2 = r1 + width;
    float* r3 = r2 + width;

    float* outptr1 = dst;
    float* outptr2 = dst + OutputWidth;

    float32x4_t k012 = vld1q_f32(kernel);
    float32x4_t k345 = vld1q_f32(kernel + 3);
    float32x4_t k678 = vld1q_f32(kernel + 6);

    k012 = vsetq_lane_f32(0.f, k012, 3);
    k345 = vsetq_lane_f32(0.f, k345, 3);
    k678 = vsetq_lane_f32(0.f, k678, 3);
    float32x4_t sum1;
    float32x4_t sum2;
    int i = 0;
    for(; i + 1 < OutputHeight; i+=2)
    {
        int remain = OutputWidth;
        for(; remain > 0; --remain)
        {
            
            float32x4_t r00 = vld1q_f32(r0);
            r00 = vsetq_lane_f32(0.f, r00, 3);
            float32x4_t r10 = vld1q_f32(r1);
            r10 = vsetq_lane_f32(0.f, r10, 3);
            float32x4_t r20 = vld1q_f32(r2);
            r20 = vsetq_lane_f32(0.f, r20, 3);
            float32x4_t r30 = vld1q_f32(r3);
            r30 = vsetq_lane_f32(0.f, r30, 3);

            //乘法版
            // sum1 = vmulq_f32(r00, k012);
            // sum1 = vmlaq_f32(sum1, r10, k345);
            // sum1 = vmlaq_f32(sum1, r20, k678);

            // sum2 = vmulq_f32(r10, k012);
            // sum2 = vmlaq_f32(sum2, r20, k345);
            // sum2 = vmlaq_f32(sum2, r30, k678);

            //加法版
            sum1 = vaddq_f32(r00, r10);
            sum1 = vaddq_f32(sum1, r20);

            sum2 = vaddq_f32(r10, r20);
            sum2 = vaddq_f32(sum2, r30);

            float32x2_t ss1 = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
            float32x2_t ss2 = vadd_f32(vget_low_f32(sum2), vget_high_f32(sum2));

            float32x2_t sss = vpadd_f32(ss1, ss2);

            *outptr1 = vget_lane_f32(sss, 0);
            *outptr2 = vget_lane_f32(sss, 1);
            
            r0++;
            r1++;
            r2++;
            r3++;

            outptr1++;
            outptr2++;
        }
        
        r0 += 2 + width;
        r1 += 2 + width;
        r2 += 2 + width;
        r3 += 2 + width;

        outptr1 += OutputWidth;
        outptr2 += OutputWidth;
    }

    for(; i < OutputHeight; i++)
    {
        int remain = OutputWidth;
        for(; remain > 0; --remain)
        {
            float32x4_t r00 = vld1q_f32(r0);
            float32x4_t r10 = vld1q_f32(r1);
            float32x4_t r20 = vld1q_f32(r2);

            float32x4_t sum1;

            sum1 = vmulq_f32(r00, k012);
            sum1 = vmlaq_f32(sum1, r10, k345);
            sum1 = vmlaq_f32(sum1, r20, k678);

            float32x2_t ss = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
            float32x2_t sss = vpadd_f32(ss, ss);

            *outptr1 = vget_lane_f32(sss, 0);

            r0++;
            r1++;
            r2++;
            outptr1++;
        }
        r0 += 2;
        r1 += 2;
        r2 += 2;
    }
}

void boxFilterConvolutionDepthwise_assem(float* src, float* dst, int width, int height, int radius)
{
    int OutputWidth = width - radius + 1;
    int OutputHeight = height - radius + 1;

    float* kernel = new float[radius * radius];
    for(int i = 0; i < radius * radius; i++)
    {
        kernel[i] = 1.0;
    }

    float32x4_t k012 = vld1q_f32(kernel);
    float32x4_t k345 = vld1q_f32(kernel + 3);
    float32x4_t k678 = vld1q_f32(kernel + 6);  //notice! there is a risk of reading out of bounds here! 

    k012 = vsetq_lane_f32(0.f, k012, 3);
    k345 = vsetq_lane_f32(0.f, k345, 3);
    k678 = vsetq_lane_f32(0.f, k678, 3);

    float* r0 = src;
    float* r1 = r0 + width;
    float* r2 = r1 + width;
    float* r3 = r2 + width;

    float* outptr1 = dst;
    float* outptr2 = dst + OutputWidth;

    int i = 0;
    for(; i + 1 < OutputHeight; i += 2)
    {
        int block = OutputWidth >> 2;
        int remain = OutputWidth - (block << 2); 

        if(block > 0)
        {
            __asm__ volatile(
                "prfm       pldl1keep,   [%0, #192]     \n"
                "ld1        {v3.4s, v4.4s}, [%0]        \n" 
                "add        %0,  %0,       #16          \n" // a b c d e f  -> 偏移到e
                "ext        v5.16b, v3.16b, v4.16b, #4  \n" //v3[a, b, c, d] v4[e, f, g, h]  v5 [b, c, d, f]
                "ext        v6.16b, v3.16b, v4.16b, #8  \n" //v6[c, d, e, f]
                
                "0:                                     \n"
                // v3 [a, b, c, d]  v5 [b, c, d, f] v6 [c, d, e, f]
                // [a, b, c, d] * k012[0]
                // [b, c, d, e] * k012[1]
                // [c, d, e, f] * k012[2]
                "fmul       v7.4s, v3.4s, %14.s[0]      \n"
                "fmul       v8.4s, v5.4s, %14.s[1]      \n"
                "fmul       v9.4s, v6.4s, %14.s[2]      \n"

                //the outptr1 second line
                "prfm       pldl1keep,       [%1, #192] \n"
                "ld1        {v3.4s, v4.4s}, [%1]        \n"
                "add        %1, %1, #16                 \n"
                "ext        v5.16b, v3.16b, v4.16b, #4  \n"
                "ext        v6.16b, v3.16b, v4.16b, #8  \n"
                "fmla       v7.4s, v3.4s, %15.s[0]     \n"
                "fmla       v8.4s, v5.4s, %15.s[1]     \n"
                "fmla       v9.4s, v6.4s, %15.s[2]     \n"

                //the outptr2 first line
                "fmul       v10.4s, v3.4s, %14.s[0]     \n"
                "fmul       v11.4s, v5.4s, %14.s[1]     \n"
                "fmul       v12.4s, v6.4s, %14.s[2]     \n"
                

                //the outptr1 third line
                "prfm       pldl1keep,        [%2, #192]\n"
                "ld1        {v3.4s, v4.4s},     [%2]    \n"
                "add        %2, %2, #16                 \n"
                "ext        v5.16b, v3.16b, v4.16b, #4  \n"
                "ext        v6.16b, v3.16b, v4.16b, #8  \n"
                "fmla       v7.4s, v3.4s, %16.s[0]     \n"
                "fmla       v8.4s, v5.4s, %16.s[1]     \n"
                "fmla       v9.4s, v6.4s, %16.s[2]     \n"

                //the outptr2 second line
                "fmla       v10.4s, v3.4s, %15.s[0]     \n"
                "fmla       v11.4s, v5.4s, %15.s[1]     \n"
                "fmla       v12.4s, v6.4s, %15.s[2]     \n"

                //the outptr2 thrid line
                "prfm       pldl1keep,      [%3, #192]  \n"
                "ld1        {v3.4s, v4.4s}, [%3]        \n"
                "add        %3, %3, #16                 \n"
                "ext        v5.16b, v3.16b, v4.16b, #4  \n"
                "ext        v6.16b, v3.16b, v4.16b, #8  \n"
                "fmla       v10.4s, v3.4s,  %16.s[0]    \n"
                "fmla       v11.4s, v5.4s,  %16.s[1]    \n"
                "fmla       v12.4s, v6.4s,  %16.s[2]    \n"
                
                "fadd       v7.4s, v7.4s, v8.4s         \n"  //outptr1

                "prfm       pldl1keep,      [%0, #192]  \n"
                "ld1        {v3.4s, v4.4s}, [%0]        \n"

                "fadd       v7.4s, v7.4s, v9.4s         \n"  //outptr1

                "fadd       v10.4s, v10.4s, v11.4s      \n"//outptr2
                "fadd       v10.4s, v10.4s, v12.4s      \n"//outptr2

                "ext        v5.16b, v3.16b, v4.16b, #4  \n"
                "ext        v6.16b, v3.16b, v4.16b, #8  \n"  
                "add        %0, %0, #16                 \n"

                "st1        {v7.4s},        [%4], #16   \n"
                "st1        {v10.4s},       [%5], #16   \n"
                
                "subs       %w6,    %w6,    #1          \n"
                "bne        0b                          \n"

                "sub        %0,     %0,     #16         \n"

                
                : "=r"(r0), //%0
                "=r"(r1),   //%1
                "=r"(r2),   //%2
                "=r"(r3),   //%3
                "=r"(outptr1),//%4
                "=r"(outptr2),//%5
                "=r"(block) //%6
                : "0"(r0),
                "1"(r1),
                "2"(r2),
                "3"(r3),
                "4"(outptr1),
                "5"(outptr2),
                "6"(block),
                "w"(k012),//%14
                "w"(k345),//%15
                "w"(k678)//%16
                :"cc", "memory", "v3", "v4", "v5","v6","v7","v8","v9","v10","v11","v12"

            );
        }

        for(; remain > 0; remain--){
            float32x4_t r00 = vld1q_f32(r0);
            float32x4_t r10 = vld1q_f32(r1);
            float32x4_t r20 = vld1q_f32(r2);
            float32x4_t r30 = vld1q_f32(r3);
            //sum1
            float32x4_t sum1 = vmulq_f32(r00, k012);
            sum1 = vmlaq_f32(sum1, r10, k345);
            sum1 = vmlaq_f32(sum1, r20, k678);

            //sum2
            float32x4_t sum2 = vmulq_f32(r10, k012);
            sum2 = vmlaq_f32(sum2, r20, k345);
            sum2 = vmlaq_f32(sum2, r30, k678);

            //[a,b,c,d]->[a,c]+[b,d]->[a+b,c+d]
            float32x2_t _ss = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
            //[e,f,g,h]->[e,g]+[f,h]->[e+f,g+h]
            float32x2_t _ss2 = vadd_f32(vget_low_f32(sum2), vget_high_f32(sum2));
            //[a+b+c+d,e+f+g+h]
            float32x2_t _sss2 = vpadd_f32(_ss, _ss2);

            *outptr1 = vget_lane_f32(_sss2, 0);
            *outptr2 = vget_lane_f32(_sss2, 1);
            
            r0++;
            r1++;
            r2++;
            r3++;
            outptr1++;
            outptr2++;
        }
        
        r0 += 2 + width;
        r1 += 2 + width;
        r2 += 2 + width;
        r3 += 2 + width;

        outptr1 += OutputWidth;
        outptr2 += OutputWidth;
    }

    for(;i < OutputHeight;i++)
    {
        int block = OutputWidth >> 2;
        int remain = OutputWidth - (block << 2);

        if(block > 0)
        {
            __asm__ volatile(
                "prfm           pldl1keep,        [%0, #192]    \n"
                "ld1            {v3.4s, v4.4s},   [%0]          \n"
                "ext            v5.16b, v3.16b, v4.16b, #4      \n"
                "ext            v6.16b, v3.16b, v4.16b, #4      \n"
                "add            %0, %0, #16                     \n"

                "0:                                             \n"
                //the first line  * k00 ~ k02
                "fmul           v7.4s,  v3.4s,  %10.s[0]        \n"
                "fmul           v8.4s,  v5.4s,  %10.s[1]        \n"
                "fmul           v9.4s,  v6.4s,  %10.s[2]        \n"

                //the second line  * k10 ~ k12
                "prfm           pldl1keep,      [%1,#192]       \n"
                "ld1            {v3.4s, v4.4s},  [%1]            \n"
                "add            %1, %1, #16                     \n"
                "ext            v5.16b, v3.16b, v4.16b, #4      \n"
                "ext            v6.16b, v3.16b, v4.16b, #4      \n"
                "fmla           v7.4s,  v3.4s,  %11.s[0]        \n"
                "fmla           v8.4s,  v5.4s,  %11.s[1]        \n"
                "fmla           v9.4s,  v6.4s,  %11.s[2]        \n"

                //the third line * k20 ~ k22
                "prfm           pldl1keep,      [%2,#192]       \n"
                "ld1            {v3.4s, v4.4s},  [%2]            \n"
                "add            %2, %2, #16                     \n"
                "ext            v5.16b, v3.16b, v4.16b, #4      \n"
                "ext            v6.16b, v3.16b, v4.16b, #4      \n"
                "fmla           v7.4s,  v3.4s,  %12.s[0]        \n"
                "fmla           v8.4s,  v5.4s,  %12.s[1]        \n"
                "fmla           v9.4s,  v6.4s,  %12.s[2]        \n"

                "fadd           v7.4s, v7.4s, v8.4s             \n"
                "fadd           v7.4s, v7.4s, v9.4s             \n"

                "st1           {v7.4s},        [%3], #16       \n"

                "prfm           pldl1keep,        [%0, #192]    \n"
                "ld1            {v3.4s, v4.4s},   [%0]          \n"
                "ext            v5.16b, v3.16b, v4.16b, #4      \n"
                "ext            v6.16b, v3.16b, v4.16b, #4      \n"
                "add            %0, %0, #16                     \n"

                "subs           %w4, %w4, #1                    \n"
                "bne            0b                              \n"
                "sub            %0, %0, #16                     \n"

                : "=r"(r0), //%0
                "=r"(r1),   //%1
                "=r"(r2),   //%2
                "=r"(outptr1),//%3
                "=r"(block) //%4
                : "0"(r0),
                "1"(r1),
                "2"(r2),
                "3"(outptr1),
                "4"(block),
                "w"(k012),//%10
                "w"(k345),//%11
                "w"(k678)//%12
                :"cc", "memory", "v3", "v4", "v5","v6","v7","v8","v9"
            );
        }

        for(; remain > 0; --remain)
        {
            float32x4_t r00 = vld1q_f32(r0);
            float32x4_t r10 = vld1q_f32(r1);
            float32x4_t r20 = vld1q_f32(r2);

            float32x4_t sum = vmulq_f32(r00, k012);
            sum = vmlaq_f32(sum, r10, k345);
            sum = vmlaq_f32(sum, r20, k678);
            
            float32x2_t res = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
            res = vpadd_f32(res, res);

            *outptr1 = vget_lane_f32(res, 0);

            r0++;
            r1++;
            r2++;
            outptr1++;
        }
        r0 += 2 + width;
        r1 += 2 + width;
        r2 += 2 + width;

        outptr1 += OutputWidth;
    }
}

void BoxFilterBetterOrigin(float *Src, float *Dest, int Width, int Height, int Radius){
    int OutWidth = Width - Radius + 1;
    int OutHeight = Height - Radius + 1;
    float *kernel = new float[Radius*Radius];
    for(int i = 0; i < Radius*Radius; i++){
        kernel[i] = 1.0;
    }
    float *k0 = kernel;
    float *k1 = kernel + 3;
    float *k2 = kernel + 6;
    float* r0 = Src;
    float* r1 = Src + Width;
    float* r2 = Src + Width * 2;
    float* r3 = Src + Width * 3;
    float* outptr = Dest;
    float* outptr2 = Dest + OutWidth;
    int i = 0;
    for (; i + 1 < OutHeight; i += 2)
    {
        int remain = OutWidth;
        for(; remain > 0; remain--){
            float sum1 = 0, sum2 = 0;
            sum1 += r0[0] * k0[0];
            sum1 += r0[1] * k0[1];
            sum1 += r0[2] * k0[2];
            sum1 += r1[0] * k1[0];
            sum1 += r1[1] * k1[1];
            sum1 += r1[2] * k1[2];
            sum1 += r2[0] * k2[0];
            sum1 += r2[1] * k2[1];
            sum1 += r2[2] * k2[2];

            sum2 += r1[0] * k0[0];
            sum2 += r1[1] * k0[1];
            sum2 += r1[2] * k0[2];
            sum2 += r2[0] * k1[0];
            sum2 += r2[1] * k1[1];
            sum2 += r2[2] * k1[2];
            sum2 += r3[0] * k2[0];
            sum2 += r3[1] * k2[1];
            sum2 += r3[2] * k2[2];
            *outptr = sum1;
            *outptr2 = sum2;
            r0++;
            r1++;
            r2++;
            r3++;
            outptr++;
            outptr2++;
        }

        r0 += 2 + Width;  // r0[0] = src[src_width - 3]  r0[1] = src[src_width -2]  r0[3] = src[src_width - 1], 因此要 + 2 才到 src 的 最边上
        r1 += 2 + Width;
        r2 += 2 + Width;
        r3 += 2 + Width;

        outptr += OutWidth;
        outptr2 += OutWidth;
    }

    for(; i < OutHeight; i++)
    {
        int remain = OutWidth;
        for(; remain > 0; remain--){
            float sum1 = 0;
            sum1 += r0[0] * k0[0];
            sum1 += r0[1] * k0[1];
            sum1 += r0[2] * k0[2];
            sum1 += r1[0] * k1[0];
            sum1 += r1[1] * k1[1];
            sum1 += r1[2] * k1[2];
            sum1 += r2[0] * k2[0];
            sum1 += r2[1] * k2[1];
            sum1 += r2[2] * k2[2];
            *outptr = sum1;
            r0++;
            r1++;
            r2++;
            outptr++;
        }

        r0 += 2;
        r1 += 2;
        r2 += 2;
    }

}

void BoxFilterBetterNeonIntrinsics(float *Src, float *Dest, int Width, int Height, int Radius){
    int OutWidth = Width - Radius + 1;
    int OutHeight = Height - Radius + 1;
    // 这里虽然 kernel 大小是根据输入设置
    // 但是下面的计算写死了是3x3的kernel
    // boxfilter 权值就是1，直接加法即可，
    // 额外的乘法会增加耗时
    float *kernel = new float[Radius*Radius];
    for(int i = 0; i < Radius*Radius; i++){
        kernel[i] = 1.0;
    }
    // 下面代码，把 kernel 的每一行存一个 q 寄存器
    // 而因为一个 vld1q 会加载 4 个浮点数，比如 k012
    // 会多加载下一行的一个数字，所以下面 
    // 会用 vsetq_lane_f32 把最后一个数字置0
    float32x4_t k012 = vld1q_f32(kernel);
    float32x4_t k345 = vld1q_f32(kernel + 3);
    // 这里 kernel 的空间如果 Radius 设为3
    // 则长度为9，而从6开始读4个，最后一个就读
    // 内存越界了，可能会有潜在的问题。
    float32x4_t k678 = vld1q_f32(kernel + 6);

    k012 = vsetq_lane_f32(0.f, k012, 3);
    k345 = vsetq_lane_f32(0.f, k345, 3);
    k678 = vsetq_lane_f32(0.f, k678, 3);

    // 输入需要同时读4行
    float* r0 = Src;
    float* r1 = Src + Width;
    float* r2 = Src + Width * 2;
    float* r3 = Src + Width * 3;
    float* outptr = Dest;
    float* outptr2 = Dest + OutWidth;
    int i = 0;
    // 同时计算输出两行的结果
    for (; i + 1 < OutHeight; i += 2){
        int remain = OutWidth;
        for(; remain > 0; remain--){
            // 从当前输入位置连续读取4个数据
            float32x4_t r00 = vld1q_f32(r0);
            float32x4_t r10 = vld1q_f32(r1);
            float32x4_t r20 = vld1q_f32(r2);
            float32x4_t r30 = vld1q_f32(r3);

            // 因为 Kernel 最后一个权值置0，所以相当于是
            // 在计算一个 3x3 的卷积点乘累加中间结果
            // 最后的 sum1 中的每个元素之后还需要再加在一起
            // 还需要一个 reduce_sum 操作
            float32x4_t sum1 = vmulq_f32(r00, k012);
            sum1 = vmlaq_f32(sum1, r10, k345);
            sum1 = vmlaq_f32(sum1, r20, k678);

            // 同理计算得到第二行的中间结果
            float32x4_t sum2 = vmulq_f32(r10, k012);
            sum2 = vmlaq_f32(sum2, r20, k345);
            sum2 = vmlaq_f32(sum2, r30, k678);

            // [a,b,c,d]->[a+b,c+d]
            // 累加 这里 vadd 和下面的 vpadd 相当于是在做一个 reduce_sum
            float32x2_t _ss = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
            // [e,f,g,h]->[e+f,g+h]
            float32x2_t _ss2 = vadd_f32(vget_low_f32(sum2), vget_high_f32(sum2));
            // [a+b+c+d,e+f+g+h]
            // 这里因为 intrinsic 最小的单位是 64 位，所以用 vpadd_f32 把第一行和第二行最后结果拼在一起了
            float32x2_t _sss2 = vpadd_f32(_ss, _ss2);
            // _sss2第一个元素 存回第一行outptr
            *outptr = vget_lane_f32(_sss2, 0);
            *outptr2 = vget_lane_f32(_sss2, 1);
            
            //同样这样直接读4个数据，也会有读越界的风险
            r0++;
            r1++;
            r2++;
            r3++;
            outptr++;
            outptr2++;
        }
        
        r0 += 2 + Width;
        r1 += 2 + Width;
        r2 += 2 + Width;
        r3 += 2 + Width;

        outptr += OutWidth;
        outptr2 += OutWidth;
    }

    for(; i < OutHeight; i++){
        int remain = OutWidth;
        for(; remain > 0; remain--){
            float32x4_t r00 = vld1q_f32(r0);
            float32x4_t r10 = vld1q_f32(r1);
            float32x4_t r20 = vld1q_f32(r2);

            //sum1
            float32x4_t sum1 = vmulq_f32(r00, k012);
            sum1 = vmlaq_f32(sum1, r10, k345);
            sum1 = vmlaq_f32(sum1, r20, k678);

            float32x2_t _ss = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
            _ss = vpadd_f32(_ss, _ss);

            *outptr = vget_lane_f32(_ss, 0);

            r0++;
            r1++;
            r2++;
            outptr++;
        }

        r0 += 2;
        r1 += 2;
        r2 += 2;
    }
}