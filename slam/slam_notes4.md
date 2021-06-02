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

### 第9讲  后端

[视觉SLAM十四讲|第10讲 后端1](https://zhuanlan.zhihu.com/p/65666168)

[扩展卡尔曼滤波EKF与多传感器融合](https://blog.csdn.net/Young_Gy/article/details/78468153)

[一文理清卡尔曼滤波，从传感器数据融合开始谈起](https://blog.csdn.net/LoseInVain/article/details/90340087)


**KF与EKF**



后端：从带噪声的数据估计内在状态 ———— 状态估计问题

渐进式估计：保持当前状态的估计，加入新信息时，更新已有估计（滤波）

批量式估计：给定一定规模的数据，计算该数据下的最优估计


线高卡，非高拓

在线性模型、高斯状态分布下，可以得到卡尔曼滤波器

在非线性模型、高斯状态分布下，可以在工作点附近线性展开，得到扩展卡尔曼滤波器

线性模型，高斯分布的线性组合，仍然是高斯分布


线性模型，高斯分布的线性组合，仍然是高斯分布，高斯分布的线性组合性质

无参情况下，粒子滤波器

后验:过了观测之后的分布

先验：推导出来的分布

预测：k-1时刻的后验通过运动方程推算K时刻的先验。

使用k-1时刻的后验分布估计k时刻的先验分布，这一步不确定性变大

第二步使用k时刻的先验分布估计k时刻的后验分布，对于结果进行修正，缩小不确定性


P是状态估计值的方差，作为系统不确定性的度量。调整卡尔曼增益，使得更新后的状态值误差的协方差最小

卡尔曼滤波更新式，通过先验推导后验

已知：运动方程、观测方程

卡5公式:

k-1后验 ---> K先验（均值、方差）

计算增益

k先验 --->  k后验（均值、方差）


增益K调整先验的（预测结果转换的）和实际观测的比例
 
卡尔曼滤波器也称为传感器融合算法


扩展卡尔曼滤波，工作点附近一阶泰勒展开，假设工作点附近是线性系统


**BA与图优化**

BA纯优化问题

属于批量式的优化方法，给定很多相机位姿和观测数据，计算最优的状态估计

定义每个运动/观测方程的误差，并从初始估计开始寻找梯度下降

BA可以用图模型清晰表达出来

顶点为优化变量，边为运动/观测约束

本身还有一些特殊的结果

每个观测只关系两个变量，其中一个是相机，一个是路标


在位姿i对路标j的一次观测的误差

图模型与H矩阵存在对应关系，图模型中存在边--> H相应对方出现非零块

箭头形：对角矩阵





