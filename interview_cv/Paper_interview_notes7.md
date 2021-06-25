### 1 柱面投影原理 

[柱面投影介绍与python实现（一）](https://blog.csdn.net/zwx1995zwx/article/details/81005454)

柱面正投影是指将平面图像投影到柱面表面的过程，柱面反投影是将柱面表面的某个特定的观察区域投影到柱面的切平面上的过程。柱面投影法是柱面全景图生成和显示过程中的必要环节。

柱面上的坐标(x',y')与平面上的坐标(x,y)的正投影关系为（维基百科公式）
	
	a = arctan(x/r)
	
	x'= r * a            俯视图
	 
	y'= y * cos a        侧视图
	
	反投影：
	
	a = arctan(x/r)
	
	x = r * tan(x’/r)
	
	y = y' / cos a
	
	r = f 相机的焦距
	
``` python	
from skimage.io import imread, imshow ,imsave
from skimage.transform import resize
import math
import numpy as np
 
img = imread('img.jpg')
img = (resize(img , [1000,800])*255).astype(np.uint8)
 
#圆柱投影
def cylindrical_projection(img , f) :
   rows = img.shape[0]
   cols = img.shape[1]
   
   #f = cols / (2 * math.tan(np.pi / 8))
 
   blank = np.zeros_like(img)
   center_x = int(cols / 2)
   center_y = int(rows / 2)
   
   for  y in range(rows):
       for x in range(cols):
           theta = math.atan((x- center_x )/ f)
           point_x = int(f * math.tan( (x-center_x) / f) + center_x)
           point_y = int( (y-center_y) / math.cos(theta) + center_y)
           
           if point_x >= cols or point_x < 0 or point_y >= rows or point_y < 0:
               pass
           else:
               blank[y , x, :] = img[point_y , point_x ,:]
   return blank
 
waved_img = cylindrical_projection(img,500)
imshow(waved_img)

程序中对三个公式的应用是反的。这是因为程序中两个for循环中的x与y对应的是柱面投影的曲面，point_x 与point_y对应的是原图中的坐标点。	
```

代码解析：
	
	 
      读取图像 fx fy cx cy  pitch roll yaw
        |
        |
      pitch roll校正（计算图像四个边界点对应的映射之后的点(坐标系转换：img --->cam 进行pitch row 变化---> img )--------> solve PnP求解对应关系）
        |
        |--------> 二维图像 柱面投影() ------>完成
	
pitch roll校正部分：

    AttToRotationEnu2Cam 根据角度计算变换矩阵

        Att2Cgb 根据欧拉角计算变换矩阵           
      |
      |-------> Image2DCode2EnuCode


柱面投影部分：	
	
    arctan((width/2) /fx))  计算角度a_w,a_h ---> 每个像素设置0.1度分辨率(角度)产生image_new，range W和H 产生alpha，beta---->根据 fx* tan(a_w)，fy*tan(a_h) 计算反投影的位置x,y--> 获得新图的位置xy
    arctan((height/2) /fy))	
	
### 2 相机标定 

[MATLAB--相机标定教程](https://blog.csdn.net/heroacool/article/details/51023921)

[实战-相机标定](https://zhuanlan.zhihu.com/p/55132750)

[相机标定——基础知识整理](https://www.guyuehome.com/7689)

[相机标定(二)——图像坐标与世界坐标转换](https://www.guyuehome.com/7832)

[相机标定(三)——手眼标定](https://www.guyuehome.com/7871)

畸变校正：可归纳如下，k1,k2,k3,k4,k5,k6径向畸变系数，p1,p2是切向畸变系数。

**相机内参**

	相机内参与镜头本身的焦距等相关，为摄像机本身特性，可通过六个参数表示为：1/dx、1/dy、s、u0、v0、f
		
	dx和dy表示x方向和y方向的一个像素分别占多少长度单位，即一个像素代表的实际物理值的大小；

	u0，v0表示图像的中心像素坐标和图像原点像素坐标之间相差的横向和纵向像素数；f为焦距；s为坐标轴倾斜参数

	在opencv文档里内参数共四个为fx、fy、u0、v0。其中fx=f*(1/dx)，fy=f*(1/dy)，s假设为0，因此为4个内参
	
	fx，fy为焦距，一般情况下，二者相等，cx、cy为主点坐标（相对于成像平面，相机光心在像素坐标系中的坐标）
	
	[ fx  0   cx
	  0   fy  cy
	  0   0   1 ]
	  
	
**相机外参**
	
相机外参矩阵描述的是相机在静态场景下相机的运动，包括旋转和平移，或者在相机固定时，运动物体的刚性运动。

相机坐标系的三个轴的旋转参数分别为（ω、δ、 θ）,然后把每个轴的3*3旋转矩阵进行组合（即先矩阵之间相乘），得到旋转矩阵R，其大小为3*3；

T的三个轴的平移参数（Tx、Ty、Tz）。R、T组合成成的3*4的矩阵。
	
	[r11  r12  r13  t1
	 r21  r22  r23  t2
	 r31  r32  r33  t3]

	 
**坐标系**

world: 世界坐标系，可以任意指定xw轴和yw​轴

camera: 相机坐标系，原点位于小孔，z轴与光轴重合，xw​轴和yw​轴平行投影面，为上图坐标系XcYcZc

image:图像坐标系，原点位于光轴和投影面的交点，xw轴和yw​轴平行投影面

pixel:像素坐标系，从小孔向投影面方向看，投影面的左上角为原点，uv轴和投影面两边重合，该坐标系与图像坐标系处在同一平面，但原点不同。

像素坐标pixel与世界坐标world的变换公式:

        Puv = KTPw

K 内参矩阵：T pixel->camera ，像素坐标系相对相机坐标系

      [ fx  0   cx
        0   fy  cy
        0   0   1 ]
	
T 外参矩阵：T camera->world，相机坐标系相对世界坐标系
	
        [r11  r12  r13  t1
         r21  r22  r23  t2
         r31  r32  r33  t3]

	
内参：通过张正友标定获得

外参：通过PNP估计获得

深度s：深度s为目标点在相机坐标系Z方向的值


### 3 抽帧的功耗问题 

	命令模式说清楚，状态机。
	
	中控器
	         -----  0 功耗优化
	        |
	-------———————— 1 检测模型 -----判断状态----reid模型
			|                          |
			 ------ 2 pose模型 -----———

### 4 模型剪枝

[模型剪枝技术原理及其发展现状和展望](https://zhuanlan.zhihu.com/p/134642289)

https://github.com/Lam1360/YOLOv3-model-pruning

稀疏化训练（1 目标方程加入稀疏正则项,batch normalization中的缩放因子γ，来约束通道的重要性）--->剪枝---> 微调

细粒度剪枝：减去连接

粗粒度剪枝：通道剪枝(Network Trimming,更高稀疏性的通道更应该被去除)

思路2：二是利用重建误差来指导剪枝，间接衡量一个通道对输出的影响，根据输入特征图的各个通道对输出特征图的贡献大小来完成剪枝过程
