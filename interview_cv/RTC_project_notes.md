###调研

RAD项目：
	
	[利用Python进行数据分析·第2版](https://www.jianshu.com/p/04d180d90a3f)
	
	[统计学习导论](https://github.com/hardikkamboj/An-Introduction-to-Statistical-Learning)
	

主板识别：

	解析卷积神经网络 
	
	resnet50、resnet18、mobilenetv3  
	
	[你必须要知道CNN模型：ResNet](https://zhuanlan.zhihu.com/p/31852747)
	
	[轻量级神经网络“巡礼”（二）—— MobileNet，从V1到V3](https://zhuanlan.zhihu.com/p/70703846)
	
	keras：EarlyStopping、HDF5数据格式
	
	[人脸识别论文再回顾之一：Center Loss](https://zhuanlan.zhihu.com/p/38235452)

OCR识别：
	
	[yolo系列之yolo v3 深度解析](https://blog.csdn.net/leviopku/article/details/82660381)
	
	[PSENET](https://zhuanlan.zhihu.com/p/37884603)
	
	[一文读懂CRNN+CTC文字识别](https://zhuanlan.zhihu.com/p/43534801)


**开源项目**

主板检测：

	https://github.com/KaiyangZhou/pytorch-center-loss


OCR检测：

	https://github.com/wizyoung/YOLOv3_TensorFlow

	https://github.com/whai362/PSENet

	图像生成：https://github.com/Belval/TextRecognitionDataGenerator

识别CRNN：

	https://github.com/MaybeShewill-CV/CRNN_Tensorflow

	https://github.com/WenmuZhou/PytorchOCR

C++读取json；

	https://github.com/nlohmann/json


###尝试方法

**数据处理**

	170类，含正反面，共板调研

	keras 旋转、对比度增强

	高斯噪声、模糊、亮度、对比度

两类数据：object365

剪枝量化：caffe 模型

OCR数据集：ICDAR2015

**CTC过程处理输出**

https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/core/util/ctc/ctc_beam_search_test.cc


**字典修正**

计算编辑距离：编辑距离查已有字典，校正输出结果

https://github.com/addaleax/levenshtein-sse


###经验

数据分析：

	Distance correlation 距离相关系数
	
	kdeplot核密度图

部署框架： NCNN、onnx_runtime

[浅析VOC数据集的mAP的计算过程](https://luckmoonlight.github.io/2019/02/24/mAP)

yolo：聚类生成anchors
