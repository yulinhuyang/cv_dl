###  第8讲 直接法视觉里程计

https://zhuanlan.zhihu.com/p/61883169


**光流法**

光流是一种描述像素随时间在图像之间运动的方法。

灰度不变假设：同一个空间点的像素灰度值，在各个图像中是固定不变的。

可以看出最小化像素误差的非线性优化

研究光流场的目的就是为了从序列图像中近似计算不能直接得到的运动场。光流场在理想情况下，光流场对应于运动场。

运动较大时使用金字塔

一言以概之：所谓光流就是瞬时速率，在时间间隔很小（比如视频的连续前后两帧之间）时，也等同于目标点的位移

三言以概之：所谓光流场就是很多光流的集合。

                     当我们计算出了一幅图片中每个图像的光流，就能形成光流场。

                     构建光流场是试图重现现实世界中的运动场，用以运动分析。
                     
LK光流法在原先的光流法两个基本假设的基础上，增加了一个“空间一致”的假设，即所有的相邻像素有相似的行动。也即在目标像素周围m×m的区域内，每个像素均拥有相同的光流矢量。

[计算机视觉--光流法(optical flow)简介](https://blog.csdn.net/qq_41368247/article/details/82562165)

[总结：光流--LK光流--基于金字塔分层的LK光流--中值流](https://blog.csdn.net/sgfmby1994/article/details/68489944)

光流金字塔：

从粗到精，上层金字塔一个像素代表下层两个像素。

每个点的光流的计算都基于----> 领域内所有点的匹配误差和最小化。

首先，从顶层开始计算金字塔最顶层图像的光流，然后根据最顶层（Lm-1）光流计算结果估计次顶层光流的初始化值。

用于跟踪图像中稀疏关键点的运动轨迹

每次使用Taylor 一阶近似，往往需要迭代多次

**稀疏光流**

稀疏光流并不对图像的每个像素点进行逐点计算。它通常需要指定一组点进行跟踪，这组点最好具有某种明显的特性，例如Harris角点等，那么跟踪就会相对稳定和可靠。

useLK.cpp

```cpp 
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
using namespace std; 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

int main( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"usage: useLK path_to_dataset"<<endl;
        return 1;
    }
    string path_to_dataset = argv[1];
    string associate_file = path_to_dataset + "/associate.txt";
    
    ifstream fin( associate_file );
    if ( !fin ) 
    {
        cerr<<"I cann't find associate.txt!"<<endl;
        return 1;
    }
    
    string rgb_file, depth_file, time_rgb, time_depth;
    list< cv::Point2f > keypoints;      // 因为要删除跟踪失败的点，使用list
    cv::Mat color, depth, last_color;
    
    for ( int index=0; index<100; index++ )
    {
        fin>>time_rgb>>rgb_file>>time_depth>>depth_file;
        color = cv::imread( path_to_dataset+"/"+rgb_file );
        depth = cv::imread( path_to_dataset+"/"+depth_file, -1 );
        if (index ==0 )
        {
            // 对第一帧提取FAST特征点
            vector<cv::KeyPoint> kps;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect( color, kps );
            for ( auto kp:kps )
                keypoints.push_back( kp.pt );
            last_color = color;
            continue;
        }
        if ( color.data==nullptr || depth.data==nullptr )
            continue;
        // 对其他帧用LK跟踪特征点
        vector<cv::Point2f> next_keypoints; 
        vector<cv::Point2f> prev_keypoints;
        for ( auto kp:keypoints )
            prev_keypoints.push_back(kp);
        vector<unsigned char> status;
        vector<float> error; 
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        cv::calcOpticalFlowPyrLK( last_color, color, prev_keypoints, next_keypoints, status, error );
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
        cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
        // 把跟丢的点删掉
        int i=0; 
        for ( auto iter=keypoints.begin(); iter!=keypoints.end(); i++)
        {
            if ( status[i] == 0 )
            {
                iter = keypoints.erase(iter);
                continue;
            }
            *iter = next_keypoints[i];
            iter++;
        }
        cout<<"tracked keypoints: "<<keypoints.size()<<endl;
        if (keypoints.size() == 0)
        {
            cout<<"all keypoints are lost."<<endl;
            break; 
        }
        // 画出 keypoints
        cv::Mat img_show = color.clone();
        for ( auto kp:keypoints )
            cv::circle(img_show, kp, 10, cv::Scalar(0, 240, 0), 1);
        cv::imshow("corners", img_show);
        cv::waitKey(0);
        last_color = color;
    }
    return 0;
}


```


**直接法**

光流法只估计了像素间的平移，

没有考虑相机本身的几何结构,没有考虑相机的旋转和图像的缩放,直接法则考虑了这些信息。

SLAM按误差函数的形式可以分为两类：最小化光度误差；最小化重投影误差。

最小化光度误差也称直接法，误差函数的形式是两个像素的灰度值相减；

最小化重投影误差也称特征点法，误差函数的形式是像素的坐标值相减。

待估计量为相机运动，关心误差对于相机的导数


稀疏直接法： 只处理稀疏角点或关键点

稠密直接法：使用所有像素

半稠密直接法：使用部分梯度明显的像素


g2o:  以顶点表示优化变量，以边表示观测方程


https://github.com/gaoxiang12/slambook/blob/master/ch8/directMethod/direct_sparse.cpp

https://github.com/gaoxiang12/slambook/blob/master/ch8/directMethod/direct_semidense.cpp


直接法的优点是：快，只要求像素梯度，能构建半稠密或稠密地图；缺点是：图像具有非凸性，单个像素区分度低需要计算图像块，灰度值不变假设可能不成立。

直接法都是就是像素，没有办法回环检测。













