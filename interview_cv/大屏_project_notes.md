### 调研

#### 论文材料 

yolov4： YOLOv4: Optimal Speed and Accuracy of Object Detection

人体检测投影：https://www.cnblogs.com/liekkas0626/p/5262942.html

hand数据集：https://blog.csdn.net/tianshixingyu/article/details/81127737

**手势（人体）关键点检测：**

https://github.com/HRNet/HRNet-Human-Pose-Estimation


**MOT跟踪：**

SORT：https://github.com/abewley/sort

[基于深度学习的目标跟踪sort与deep-sort](https://blog.csdn.net/XSYYMY/article/details/81747134)

https://github.com/nwojke/deep_sort

实现参考： https://github.com/mcximing/sort-cpp


**MOT跟踪：**

SORT：https://github.com/abewley/sort

[基于深度学习的目标跟踪sort与deep-sort](https://blog.csdn.net/XSYYMY/article/details/81747134)

https://github.com/nwojke/deep_sort

实现参考： https://github.com/mcximing/sort-cpp


**人脸关键点检测 PFLD： 300W-LP**

[68人脸关键点](https://blog.csdn.net/u013841196/article/details/85720897)

https://github.com/ainrichman/Peppa-Facial-Landmark-PyTorch


唇动检测；

https://github.com/sachinsdate/lip-movement-net

#### 开源项目：

目标检测

yolov3

https://github.com/ultralytics/yolov3

YOLOv3-model-pruning on oxford hand 

https://github.com/Lam1360/YOLOv3-model-pruning

https://github.com/eric612/MobileNet-YOLO

yolov4

https://github.com/AlexeyAB/darknet

https://github.com/Tianxiaomo/pytorch-YOLOv4

https://github.com/tanluren/yolov3-channel-and-layer-pruning

yolov5:

https://github.com/ultralytics/yolov5



### 方法尝试

**NNIE：**

	[如何在Hi3559A上运行自己的YOLOv3模型](https://blog.csdn.net/qq_34533248/article/details/102498143)

	[关于海思3559A V100芯片IVE算子的总结]：https://blog.wuzijian.tech/chat/2020/09/18/%E6%B5%B7%E6%80%9D%E8%8A%AF%E7%89%873559AV100%E4%BD%BF%E7%94%A8%E5%BF%83%E5%BE%97/

	[图解YU12、I420、YV12、NV12、NV21](https://blog.csdn.net/byhook/article/details/84037338)

**kalmanfilter的C++实现：**

	Gemm 函数集成：[通用矩阵乘（GEMM）优化与卷积计算](https://zhuanlan.zhihu.com/p/66958390)

	[SVD求逆C++实现](https://blog.csdn.net/fengbingchun/article/details/72853757)
	
**LSTM-->dense-->CONV1d：**

	[一维卷积解决时间序列问题](https://juejin.cn/post/6844903713224523789)

	上下嘴唇距离-->上下嘴唇比人脸中间线之后的归一化距离
	
PNP计算人脸角度：

	[由6,14以及68点人脸关键点计算头部姿态](https://blog.csdn.net/ChuiGeDaQiQiu/article/details/88623267)
	
Blaze face：更小的人脸检测器

MNN推理框架：CPU、VULKAN、GPU
	
### 成功经验

3559内核烧写、rootfs烧写、uvc驱动烧写

挥手判断逻辑：先框匹配 --> 15帧内，框中心移动的距离,左超过阈值，右超过阈值

一维卷积、归一化距离（比上人脸中间线）

人脸检测-->关键点检测-->唇动识别

联合体：https://stackoverflow.com/questions/18270974/how-to-convert-a-float-to-a-4-byte-char-in-c




