# arm-neon-boxfilter
- mkdir build
- cd build
- cmake ../
- make

测试平台nvidia jetson agx，armv8.2a，1.2GHZ, 使用单核。

图像分辨率为：6000*4000 ，半径 3， 循环次数为 1。 优化等级 -O0 不优化，cache数据使用 valgrind工具集中的 cachegrind得到的。

|            函数            | 时间 ms |    I   refs    |    D  refs     | D1 misses  | D1 miss rate |
| :------------------------: | :-----: | :------------: | :------------: | :--------: | :----------: |
|      naive_boxFilter       |   517   | 31,309,575,754 | 15,998,052,693 | 22,293,979 |     0.1%     |
|    boxFilterOpenCVTrick    |   208   | 11,186,366,287 | 5,372,793,923  | 25,293,980 |     0.5%     |
|   boxFilterOpencvTrick2    |   57    | 5,188,540,004  | 2,133,961,806  | 61,298,472 |     2.9%     |
|   boxFilterOpencvTrick3    |   38    | 5,620,261,030  | 2,349,810,341  | 19,302,531 |     0.8%     |
|      boxFilterArmNeon      |   31    | 4,372,847,981  | 1,834,094,831  | 19,310,543 |     1.1%     |
|   boxFilterArmNeon_assem   |   23    | 4,031,386,423  | 1,540,555,287  | 19,320,842 |     1.3%     |
| convolutiondepthwise_naive |   42    | 5,342,628,255  | 2,528,651,127  | 14,801,973 |     0.6%     |
| ConvolutionDepthwise_neon  |   80    | 5,390,588,274  | 3,092,181,255  | 14,801,984 |     0.5%     |
| ConvolutionDepthwise_assem |    7    | 3,071,376,215  |  988,675,454   | 14,839,204 |     1.5%     |
