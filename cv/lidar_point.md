point net--> Voxel Net-->point pillars—>point RCNN --> Point Paint

point net:

[【3D视觉】PointNet和PointNet++](https://zhuanlan.zhihu.com/p/336496973)
  
  
Voxel Net:

[【3D物体检测】VoxelNet论文和代码解析](https://zhuanlan.zhihu.com/p/352419316)
  
   将点云数据纳入一个个体素立方块，构成规则的、密集分布的体素集。
  
point pillars:

[【3D目标检测】PointPillars论文和代码解析](https://zhuanlan.zhihu.com/p/357626425)

  将Point转化成一个个的Pillar（柱体），从而构成了伪图片的数据。
  
  
point RCNN: 

[论文阅读：PointRCNN 第一个基于原始点云的3D目标检测](https://zhuanlan.zhihu.com/p/71564244)

  pointnet提取点云的特征 --> 前景点分割网络 --> box生成网络
  
Point Paint:

[PointPainting: 融合图像语意分割信息到3D点云目标检测中](https://zhuanlan.zhihu.com/p/96449107)

  (1)Deeplab3进行语意分割->(2)把语意分割结果画到点云上->(3)用纯点云方法如pointpillar、PointRCNN进行3D目标检测
	
	BEV视角：BEV(Bird's Eye View) Map
	
	把语意分割信息画到点云上：把每个点云数据透过相机内外参数投影到图像座标中，在将图像中的语意分割分类的分数加到该点上。


其他： 从俯视角度将点云数据进行处理，获得一个个伪图片的数据。常见的模型有MV3D和AVOD

