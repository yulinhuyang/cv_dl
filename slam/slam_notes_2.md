主要笔记：

https://blog.csdn.net/ncepu_Chen/article/details/105322585


## slam  十四讲 notes补充

### 1-2 讲 概述

	传感器数据---> 前端 视觉里程计 --->后端非线性优化 ---> 建图

				  ---> 回环检测       --->
      
      
**视觉里程计(Visual Odometry, VO)**

估算相邻图像间相机的运动，以及局部地图的样子。

视觉里程计就是小萝卜走一步看一眼，边走边看，边看边算，把他看过的所有图片恢复成场景。这其中有一个问题叫做累积漂移(Accumulating Drift) 

**后端优化(Optimization)**

接受不同时刻视觉里程计测量的相机位姿，以及回环检测的信息，对他们进行优化，得到全局一致的轨迹和地图。

“位姿”：相机的位置（空间坐标）和姿态（角度），有视觉里程计提供。

“全局一致的轨迹和地图”：视觉里程计只关注当前看到的场景，在后端要把每一个局部的场景拼接起来，形成一个完整的场景，这就要保证全局一致性。

视觉里程计称为前端，直接对传感器数据进行加工计算，将计算结果传递到后端。这结果中很可能包含着误差，传感器的测量误差，计算误差、噪声等，需要在后端进行优化，才能保证全局一致性。

后端优化可以看作是状态估计问题，从带有噪声的数据中估计整个系统的状态，以及这个状态估计的不确定性有多大。前端与计算机视觉领域更相关，后端与滤波、非线性优化相关。

**回环检测(Loop Closing)**

判断机器人是否到达过先前的位置。如果检测到回环，它会把信息提供给后端处理。

**建图(Mapping)**

建图也就是按照需求，让小萝卜把当前的环境描述出来，

度量地图：要通过格子/方块（二维画格子，三维画方块）把空间进行划分，来精确的表示地图中的地理位置。稀疏度量地图之表示路标，稠密度量地图表示所有看到的东西。

拓扑地图：强调地图当中元素的关系，用图(Graph)表示，AB两点之间有边表示AB连通，并不考虑如何从A到达B。

**运动方程：**

运动方程在说，在k时刻，根据传感器的读数Uk 和k-1时刻的位置Xk-1，小萝卜在脑海中通过函数f()估算出当前时刻的位置是Xk。在估算的时候，小萝卜还要把噪声wk考虑进去。

**观测方程：**

观测方程在说，在k时刻，小萝卜处在Xk 的位置，此时小萝卜看见了路标点yj ，产生了一个观测结果zkj。此时小萝卜的位置由运动方程求出和观测结果已知，路标点的坐标未知。考虑噪声ukj。


### 第3讲 三维空间刚体运动

内积：夹角cos

外积：行列式

刚体：刚体是指在运动中和受力作用后，形状和大小不变，而且内部各点的相对位置不变的物体

欧式变换：同一个向量，从坐标系A到坐标系B中，长度，夹角都不变，这个坐标系从A到B的过程交欧式变换。

**特殊正交群(special orthogonal group)** 

一类元素行列式为1的重要的典型群。

旋转矩阵刻画了旋转前后同一个向量的坐标变换关系。

SO(n)={R∈Rn×n∣RRT=I,det(R)=1}


**特殊欧式群SE（special Euclidean group）**

用四个数表达三维向量的做法，称为齐次坐标。

引入齐次坐标后，旋转和平移放在同一个矩阵，称为变换矩阵。

变换矩阵T , T构成特殊欧式群SE


**Eigen实践**

官方库： https://gitlab.com/libeigen/eigen


```CPP
#include <iostream>
using namespace std;
#include <ctime>
// Eigen 部分
#include <Eigen/Core>
// 稠密矩阵的代数运算（逆，特征值等）
#include <Eigen/Dense>

#define MATRIX_SIZE 50

/****************************
* 本程序演示了 Eigen 基本类型的使用
****************************/

int main( int argc, char** argv )
{
    // Eigen 中所有向量和矩阵都是Eigen::Matrix，它是一个模板类。它的前三个参数为：数据类型，行，列
    // 声明一个2*3的float矩阵
    Eigen::Matrix<float, 2, 3> matrix_23;

    // 同时，Eigen 通过 typedef 提供了许多内置类型，不过底层仍是Eigen::Matrix
    // 例如 Vector3d 实质上是 Eigen::Matrix<double, 3, 1>，即三维向量
    Eigen::Vector3d v_3d;
	// 这是一样的
    Eigen::Matrix<float,3,1> vd_3d;

    // Matrix3d 实质上是 Eigen::Matrix<double, 3, 3>
    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero(); //初始化为零
    // 如果不确定矩阵大小，可以使用动态大小的矩阵
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > matrix_dynamic;
    // 更简单的
    Eigen::MatrixXd matrix_x;
    // 这种类型还有很多，我们不一一列举

    // 下面是对Eigen阵的操作
    // 输入数据（初始化）
    matrix_23 << 1, 2, 3, 4, 5, 6;
    // 输出
    cout << matrix_23 << endl;

    // 用()访问矩阵中的元素
    for (int i=0; i<2; i++) {
        for (int j=0; j<3; j++)
            cout<<matrix_23(i,j)<<"\t";
        cout<<endl;
    }

    // 矩阵和向量相乘（实际上仍是矩阵和矩阵）
    v_3d << 3, 2, 1;
    vd_3d << 4,5,6;
    // 但是在Eigen里你不能混合两种不同类型的矩阵，像这样是错的
    // Eigen::Matrix<double, 2, 1> result_wrong_type = matrix_23 * v_3d;
    // 应该显式转换
    Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    cout << result << endl;

    Eigen::Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
    cout << result2 << endl;

    // 同样你不能搞错矩阵的维度
    // 试着取消下面的注释，看看Eigen会报什么错
    // Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<double>() * v_3d;

    // 一些矩阵运算
    // 四则运算就不演示了，直接用+-*/即可。
    matrix_33 = Eigen::Matrix3d::Random();      // 随机数矩阵
    cout << matrix_33 << endl << endl;

    cout << matrix_33.transpose() << endl;      // 转置
    cout << matrix_33.sum() << endl;            // 各元素和
    cout << matrix_33.trace() << endl;          // 迹
    cout << 10*matrix_33 << endl;               // 数乘
    cout << matrix_33.inverse() << endl;        // 逆
    cout << matrix_33.determinant() << endl;    // 行列式

    // 特征值
    // 实对称矩阵可以保证对角化成功
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver ( matrix_33.transpose()*matrix_33 );
    cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
    cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;

    // 解方程
    // 我们求解 matrix_NN * x = v_Nd 这个方程
    // N的大小在前边的宏里定义，它由随机数生成
    // 直接求逆自然是最直接的，但是求逆运算量大

    Eigen::Matrix< double, MATRIX_SIZE, MATRIX_SIZE > matrix_NN;
    matrix_NN = Eigen::MatrixXd::Random( MATRIX_SIZE, MATRIX_SIZE );
    Eigen::Matrix< double, MATRIX_SIZE,  1> v_Nd;
    v_Nd = Eigen::MatrixXd::Random( MATRIX_SIZE,1 );

    clock_t time_stt = clock(); // 计时
    // 直接求逆
    Eigen::Matrix<double,MATRIX_SIZE,1> x = matrix_NN.inverse()*v_Nd;
    cout <<"time use in normal inverse is " << 1000* (clock() - time_stt)/(double)CLOCKS_PER_SEC << "ms"<< endl;
    
	// 通常用矩阵分解来求，例如QR分解，速度会快很多
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout <<"time use in Qr decomposition is " <<1000*  (clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;

    return 0;
}

```

**旋转矩阵**

矩阵分解： QR 分解。

**旋转向量(轴角表示法)** 

可以使用一个向量, 其方向表示旋转轴而长度表示旋转角.这种向量称为旋转向量(或轴角,Axis-Angle)

旋转向量和旋转矩阵之间的转换:  罗德里格斯公式

https://blog.csdn.net/q583956932/article/details/78933245

**欧拉角**

欧拉角将一次旋转分解成3个分离的转角.

ZYX 转角： Z -- yaw 偏航 ,Y -- pitch 俯仰 ,X -- roll 滚转

**四元数**

四元数是一种扩展的复数,既是紧凑的,也没有奇异性.

一个实部，三个虚部。

四元数与旋转角度进行转换，与旋转向量之间的转换，与角轴表示转换


```cpp

    // Eigen/Geometry 模块提供了各种旋转和平移的表示
    // 3D 旋转矩阵直接使用 Matrix3d 或 Matrix3f
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    // 旋转向量使用 AngleAxis, 它底层不直接是Matrix，但运算可以当作矩阵（因为重载了运算符）
    Eigen::AngleAxisd rotation_vector ( M_PI/4, Eigen::Vector3d ( 0,0,1 ) );     //沿 Z 轴旋转 45 度
    cout .precision(3);
    cout<<"rotation matrix =\n"<<rotation_vector.matrix() <<endl;                //用matrix()转换成矩阵
    // 也可以直接赋值
    rotation_matrix = rotation_vector.toRotationMatrix();
    // 用 AngleAxis 可以进行坐标变换
    Eigen::Vector3d v ( 1,0,0 );
    Eigen::Vector3d v_rotated = rotation_vector * v;
    cout<<"(1,0,0) after rotation = "<<v_rotated.transpose()<<endl;
    // 或者用旋转矩阵
    v_rotated = rotation_matrix * v;
    cout<<"(1,0,0) after rotation = "<<v_rotated.transpose()<<endl;

    // 欧拉角: 可以将旋转矩阵直接转换成欧拉角
    Eigen::Vector3d   = rotation_matrix.eulerAngles ( 2,1,0 ); // ZYX顺序，即roll pitch yaw顺序
    cout<<"yaw pitch roll = "<<euler_angles.transpose()<<endl;

    // 欧氏变换矩阵使用 Eigen::Isometry
    Eigen::Isometry3d T=Eigen::Isometry3d::Identity();                // 虽然称为3d，实质上是4＊4的矩阵
    T.rotate ( rotation_vector );                                     // 按照rotation_vector进行旋转
    T.pretranslate ( Eigen::Vector3d ( 1,3,4 ) );                     // 把平移向量设成(1,3,4)
    cout << "Transform matrix = \n" << T.matrix() <<endl;

    // 用变换矩阵进行坐标变换
    Eigen::Vector3d v_transformed = T*v;                              // 相当于R*v+t
    cout<<"v tranformed = "<<v_transformed.transpose()<<endl;

    // 对于仿射和射影变换，使用 Eigen::Affine3d 和 Eigen::Projective3d 即可，略

    // 四元数
    // 可以直接把AngleAxis赋值给四元数，反之亦然
    Eigen::Quaterniond q = Eigen::Quaterniond ( rotation_vector );
    cout<<"quaternion = \n"<<q.coeffs() <<endl;   // 请注意coeffs的顺序是(x,y,z,w),w为实部，前三者为虚部
    // 也可以把旋转矩阵赋给它
    q = Eigen::Quaterniond ( rotation_matrix );
    cout<<"quaternion = \n"<<q.coeffs() <<endl;
    // 使用四元数旋转一个向量，使用重载的乘法即可
    v_rotated = q*v; // 注意数学上是qvq^{-1}
    cout<<"(1,0,0) after rotation = "<<v_rotated.transpose()<<endl;


```


pangolin： slam 可视化程序


### 第5讲  相机模型




 
 
 
 
