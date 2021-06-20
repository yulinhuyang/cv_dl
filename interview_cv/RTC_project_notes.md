### 调研

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


**PSENet**

PSENet不仅适应任意角度的文本检测，而且对近距离文本分割效果更好。

任意形状的文字快：

	可以检测任意形状的文字块，很多文字检测的方法都是基于bounding box回归的，很难准确地定位弯曲文字块。

	基于语义分割的方法恰好能很好地解决这个问题，语义分割可以从像素级别上分割文字区域和背景区域。

很近的文字快：

	直接用语义分割来检测文字又会遇到新的问题：很难分离靠的很近的文字块。
	
	因为语义分割只关心每个像素的分类问题，所以即使文字块的一些边缘像素分类错误对loss的影响也不大。对于这个问题，一个直接的思路是：增大文字块之间的距离，使它们离得远一点。
	
	引入了新的概念“kernel”，顾名思义就是文字块的核心。从Fig. 2中我们可以看到：利用“kernel”可以有效地分离靠的很近的文字块。
	
通过“kernel”来构建完整的文字块：

	基于广度优先搜索的渐进扩展算法来构建完整的文字块，从每个“kernel”出发，利用广度优先搜索来不断地合并周围的像素，使得“kernel”不断地扩展，最后得到完整的文字块。
	

主干流程： 

	PSENet的主干网络是FPN--->分出4个featuremap-->concat一起(需要上采样) ————>预测不同的kernel scale S1----Sn ---> 渐进扩展算法扩张S1, 逐减扩展开

	F = C(P2,P3,P4,P5) = P2||Upx2(P3)||Upx4(P4)||Upx8(P5)
	
	P5--------              -----Sn
                 |	       |
	P4-------- concat----->F ----Sn-1   -----> 操作原图
	         |             |_____S1
	P3-------- 
	
	
	N = 7
	
	损失函数：λLc +(1−λ)Ls , sdice coefficient，Lc为文本区域分类损失，Ls为收缩文本实例损失。

	Dice loss：采用交叉熵损失会导致由负样本主导，训练难以收敛，因此训练采用dice coefficient,  X并Y/(X+Y)


yolov3 spp   backbone是mobilenet v2


**CRNN：**

	CNN ---> RNN ---> CTC

	
CNN: 降采样，将大小为（32,100,3）--> 转换为（1,25,512）feature map

	  channel:512, h：1  w:26
	  
	  切分成26个feature sequence
	  
RNN: 将26个 feature sequence 输入到RNN中，T = 26
		
	 是一个双向的LSTM
	 
	 每个时刻选择概率最大的那个，相同的时刻只取一个，然后去掉空白符
	 
CTC:
	
	CTC是一种Loss计算方法，用CTC代替Softmax Loss，训练样本无需对齐。CTC特点：

	引入blank字符，解决有些位置没有字符的问题
	
	通过递推，快速计算梯度
	 
	解码： 束搜索（Beam Search）
	
