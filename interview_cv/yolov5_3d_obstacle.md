
obstacles

KITTI数据集--参数: https://blog.csdn.net/cuichuanchen3307/article/details/80596689

https://github.com/poodarchu/Det3D

https://github.com/ultralytics/yolov5

smoker 论文和源码： https://blog.csdn.net/weixin_39326879/article/details/112298193

3D目标检测之SMOKE  https://mp.weixin.qq.com/s/sRBRJxI1hwIU0QMCajzePw#at

yolov4&v5训练代码理解   https://blog.csdn.net/cdknight_happy/article/details/109817548


数据预处理：

	图像数据
							  --->时间戳对齐 --->坐标系转换 ----> 鱼眼图像+ 对应的3点云检测结果 ----->柱面图像 + 3D点云标注结果
	点云数据（det3d）预标注

	https://github.com/poodarchu/Det3D
	
	
	label： 
	
		2D: x y w  h
		
		3D：z x y   h w l  rotation_y 

模型结构：

	P1: FocusWukong                   x(b,c,h,w) -> y(b,4c,h/2,w/2)  ---->  仿照passthrough的 focus结构
	 |     
    P2: Conv + BottleneckCSP2Light   对比原生的yolov5  BottleneckCSP去掉了 最后一个CBL结构
	|
	P3: Conv + BottleneckCSP2Light   
	|
	P4
	|
	P5：
	conv
	|_______
			Cat  backbone P4-P3
		
			Cat  head  P4-p6
			 |
			DetectSSMH 分支：人、车、信号灯、锥桶
	
	-------	Cat  backbone P4-P3
			|
			Cat  head  P4-P6  ——  检测3D方向 orient
			|
			|————————————————————— 检测3D 人 VRU (4个尺度)
			|
			|————————————————————— 检测3D车  VEH (4个尺度)
    
	detect的分支输入3层，变成4层		
			
			
	预测：  		
		拼合2D和3D的预测结果
		|
		non_max_suppression_3d 借助2D的nms的结果，筛选3D框
		|
		解码 
		
		depth： depths_offset * self.depth_ref[1]  + self.depth_ref[0] 
		
		location：预测点（featuremap维）---> 乘以trans_inv 转到img ---> 柱面反投影 ---> 乘以z  ---> 乘以K_inv  ---> decode后处理后需要乘以rot_p矩阵，校正pitch
		
		dimension： (dims_offset.exp()* dims_select  L W h)和 orientation(roty 、alpha )
		
		2D框：预测x y w h 和类别
		
		3D框：认为每个点都有3D框（xc yc的投影点），通过对应的2D框去筛选出来有效的xc yc,然后再去用3D框计算x y z w h l等。
		
loss计算：

	2D loss：lbox(GIOU loss) + lobj  + lcls(BCEcls)
	
	3D loss: lobj（BCEobj） + l3d_ori ( L1  rotys  sin(alpha)  cos(alpha)) + l3d_dim(L1  box3d_dims) + l3d_loc(L1  box3d_locs)
		
	利用2D的anchor 和target的信息 ，筛选出3D对应的layer的target。这样可以就将3D的GT 分配到了layer上
	
    
