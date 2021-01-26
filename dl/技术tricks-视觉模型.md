**1 视觉模型：**

模型：分类(bone)、检测、分割、GAN、人脸(人体、reid、度量) 、其他（NLP、OCR）
OCR参考：深度实践OCR：基于深度学习的文字识别

框架和算子：TensorFlow：实战Google深度学习框架（TeGo）、
深度学习框架PyTorch：入门与实践（DLPy）、
Python深度学习PDL、动手学深度学习DSL(Pytorch版)

API官方手册：TensorFlow、PyTorch、Numpy 、NNI


**2 深度学习理论**

主要参考：
深度学习DL、神经网络与深度学习ND
深入理解AutoML和AutoDL


3 模型

**3.1 分类：**

ResNet：  

[ResNet及其变种的结构梳理、有效性分析与代码解读](https://zhuanlan.zhihu.com/p/54289848)

[关于ResNeSt的点滴疑惑](https://zhuanlan.zhihu.com/p/133805433)

Resnext:   [ResNeXt详解](https://zhuanlan.zhihu.com/p/51075096)

Res2net

ResNeSt:   [ResNeSt：Split-Attention Networks](https://zhuanlan.zhihu.com/p/132655457)

Inception:   https://zhuanlan.zhihu.com/p/37505777

MobileNet:   

[轻量级神经网络“巡礼”（二）—— MobileNet，从V1到V3](https://zhuanlan.zhihu.com/p/70703846)

[MobileNet 详解深度可分离卷积](https://zhuanlan.zhihu.com/p/80177088)

Senet：  https://zhuanlan.zhihu.com/p/65459972

SqueezeNet：  https://zhuanlan.zhihu.com/p/31558773  

ShuffleNet:  [轻量级神经网络“巡礼”（一）—— ShuffleNetV2](https://zhuanlan.zhihu.com/p/67009992)

efficientnet： [令人拍案叫绝的EfficientNet和EfficientDet](https://zhuanlan.zhihu.com/p/96773680)

hrnet:  [打通多个视觉任务的全能Backbone:HRNet](https://zhuanlan.zhihu.com/p/134253318)


**3.2 检测：**

**Two-stage**

Faster RCNN:  

[一文读懂Faster RCNN](https://zhuanlan.zhihu.com/p/31426458)

[捋一捋pytorch官方FasterRCNN代码](https://zhuanlan.zhihu.com/p/145842317)

Mask RCNN:

[令人拍案称奇的Mask RCNN](https://zhuanlan.zhihu.com/p/37998710)

Mask score RCNN:  https://zhuanlan.zhihu.com/p/111722103

**One-stage**

SSD： [目标检测|SSD原理与实现](https://zhuanlan.zhihu.com/p/33544892)

Yolo

[你真的读懂yolo了吗？](https://zhuanlan.zhihu.com/p/37850811)

[Yolo三部曲解读——Yolov3](https://zhuanlan.zhihu.com/p/76802514)

[Yolo三部曲解读——Yolov2](https://zhuanlan.zhihu.com/p/74540100)

[深入浅出Yolo系列之Yolov3&Yolov4&Yolov5核心基础知识完整讲解](https://zhuanlan.zhihu.com/p/143747206)

[深入浅出Yolo系列之Yolov5核心基础知识完整讲解](https://zhuanlan.zhihu.com/p/172121380)

gaussion yolo

RetinaNet：  https://zhuanlan.zhihu.com/p/133317452

**Anchor-Free**

https://zhuanlan.zhihu.com/p/62103812

FCOS：  https://zhuanlan.zhihu.com/p/62869137

NAS_FPN：  https://zhuanlan.zhihu.com/p/63300940

*FPN专题*	 ： https://zhuanlan.zhihu.com/p/148738276

**mmdetetion补充**
**RCNN改进**

Cascade R-CNN：  [Cascade R-CNN 详细解读](https://zhuanlan.zhihu.com/p/42553957)

Cascade Mask R-CNN：  

Foveabox 

Hybrid Task Cascade：  [实例分割的进阶三级跳：从 Mask R-CNN 到 Hybrid Task Cascade](https://zhuanlan.zhihu.com/p/57629509)

Guided Anchoring

FreeAnchor

ATSS: [ATSS : 目标检测的自适应正负anchor选择](https://zhuanlan.zhihu.com/p/115407465)

FSAF： https://zhuanlan.zhihu.com/p/58782838

Foveabox： https://zhuanlan.zhihu.com/p/63190983

PointRend： https://zhuanlan.zhihu.com/p/98351269


**不均衡问题**

OHEM: [Hard Negative Mining/OHEM 你真的知道二者的区别吗？](https://zhuanlan.zhihu.com/p/78837273)	

GHM：  https://zhuanlan.zhihu.com/p/80594704

Generalized Focal Loss：  https://zhuanlan.zhihu.com/p/147691786	

**算子提升**

DCNv2： 	https://zhuanlan.zhihu.com/p/180075757

RepPoints： https://zhuanlan.zhihu.com/p/64522910

CARAFE：	https://zhuanlan.zhihu.com/p/76063768

Group Normalization： https://zhuanlan.zhihu.com/p/35005794

Weight Standardization： https://zhuanlan.zhihu.com/p/61783291	

Mixed Precision (FP16) Training： https://zhuanlan.zhihu.com/p/84219777

**新思考**

Soft-NMS：

https://zhuanlan.zhihu.com/p/157900024

https://zhuanlan.zhihu.com/p/41046620

Generalized Attention：



GCNet	

InstaBoost

GRoIE	

DetectoRS: [54.7 AP！最强的目标检测网络：DetectoRS（已开源）](https://zhuanlan.zhihu.com/p/145897444)

Detr 	

**3.3 跟踪**


SiamFC：  https://zhuanlan.zhihu.com/p/66757733

SiamRPN	

SiamRPN++： [视觉目标跟踪之SiamRPN++](https://zhuanlan.zhihu.com/p/56254712)

SiamMask: [我对Siamese网络的一点思考（SiamMask）](https://zhuanlan.zhihu.com/p/58154634)

SiamRCNN

SiamFC++

DIMP	

FairMOT：  [MOT开源实时新SOTA | FairMOT](https://zhuanlan.zhihu.com/p/126558285)	

CenterTrack：  https://zhuanlan.zhihu.com/p/125395219		

**3.4 分割**

**语义分割**

[语义分割中的深度学习方法全解：从FCN、SegNet到各代DeepLab](https://zhuanlan.zhihu.com/p/27794982)

[语义分割论文简析：DeepLab、GCN、DANet、PSPNet、DenseASPP、PAN...](https://zhuanlan.zhihu.com/p/75415302)

FCN： 

[全卷积网络 FCN 详解](https://zhuanlan.zhihu.com/p/30195134)

[图像语义分割入门+FCN/U-Net网络解析](https://zhuanlan.zhihu.com/p/31428783)

Unet	

Segnet： https://zhuanlan.zhihu.com/p/58888536

Pspnet：  

Deeplabv3: [DeepLab 语义分割模型 v1、v2、v3、v3+ 概要（附 Pytorch 实现）](https://zhuanlan.zhihu.com/p/68531147)

Deeplabv3plus:   [语义分割模型之DeepLabv3+](https://zhuanlan.zhihu.com/p/62261970)

PSANet 	

UPerNet: https://zhuanlan.zhihu.com/p/42982922

NonLocal Net：   

[GCNet：当Non-local遇见SENet](https://zhuanlan.zhihu.com/p/64988633)

[语义分割中的Attention和低秩重建](https://zhuanlan.zhihu.com/p/77834369)

EncNet	

CCNet: [CCNet--于"阡陌交通"处超越恺明Non-local](https://zhuanlan.zhihu.com/p/51393573)

DANet 

GCNet

ANN

OCRNet	

**实例分割**

[实例分割最新最全面综述：从Mask R-CNN到BlendMask](https://zhuanlan.zhihu.com/p/110132002)

[【进展综述】单阶段实例分割（Single Stage Instance Segmentation）](https://zhuanlan.zhihu.com/p/102231853)

FCIS: 

https://zhuanlan.zhihu.com/p/27500215

YOLACT-700	

YOLACT++： [YOLACT++：更强的实时实例分割网络，可达33.5 FPS/34.1mAP](https://zhuanlan.zhihu.com/p/97684893)

PolarMask： [PolarMask: 一阶段实例分割新思路](https://zhuanlan.zhihu.com/p/84890413)

SOLO：  [看待SOLO: Segmenting Objects by Locations，是实例分割方向吗？](https://www.zhihu.com/question/360594484/answer/936591301)

SOLOV2：

[SOLOv2: Dynamic, Faster and Stronger](https://zhuanlan.zhihu.com/p/120263670)

PointRend  [何恺明团队新作PointRend：并非神作，但的确很有意义](https://zhuanlan.zhihu.com/p/98351269)

https://zhuanlan.zhihu.com/p/98508347

BlendMask:

https://zhuanlan.zhihu.com/p/103256935

TensorMask:

https://zhuanlan.zhihu.com/p/60984659

conv instance	


**3.5 GAN**

Pix2Pix(Pix2PixHD): [图像翻译三部曲：pix2pix, pix2pixHD, vid2vid](https://zhuanlan.zhihu.com/p/56808180)

CycleGAN(StarGAN):  [CycleGAN的原理与实验详解]  https://zhuanlan.zhihu.com/p/28342644

SRGAN(ESRGAN): [SRGAN With WGAN，让超分辨率算法训练更稳定](https://zhuanlan.zhihu.com/p/37009085)

Progressing GAN: https://zhuanlan.zhihu.com/p/30532830

StyleGAN:  https://zhuanlan.zhihu.com/p/263554045

**3.6 人脸人体、  3D、Reid、度量、image retrival**

MTCNN：	

	构建图像金字塔
	P-Net 、R-Net、o-Net
	Soft-NMS:降低置信度
	5个关键点

		
RetinaFace：   https://zhuanlan.zhihu.com/p/103005911

	SSH检测模块
	关键点回归Dense Regression Branch：2D 3D映射图卷积


**人脸loss**

[人脸识别的LOSS（上）](https://zhuanlan.zhihu.com/p/34404607)

[人脸识别的LOSS（下）](https://zhuanlan.zhihu.com/p/34436551)

[人脸识别损失函数简介与Pytorch实现：ArcFace、SphereFace、CosFace](https://zhuanlan.zhihu.com/p/60747096)


Triple loss：三元组

Center loss:类别中心

主要：权值和特征归一化

Largin margin：强加分类，m倍角度

SphereFace：归一化权值W

Additive Margin Loss: 乘性变加，cos(theta) -m

ArcFace：Cos(theta+m)


hourglass

	多个pipeline分别单独处理不同尺度下的信息，再网络的后面部分再组合这些特征
	中间输出heatmap
	中间监督

**3D人脸**

[人脸重建速览，从3DMM到表情驱动动画](https://zhuanlan.zhihu.com/p/58631750)


3DMM

[【技术综述】基于3DMM的三维人脸重建技术总结](https://zhuanlan.zhihu.com/p/161828142)

Firstorder

Prnet

3DFFA

**3D人体**

SMPL

SMPL+D

PIFUHD

**Reid**

Bnneck

Gempooling

Non_local

Warmup learning

Circle loss

**Metric-learing**

FastAP

Marigin Triplet loss

CrossBatch Memory

**对比学习**

MOCO

[一文梳理无监督对比学习（MoCo/SimCLR/SwAV/BYOL/SimSiam](https://zhuanlan.zhihu.com/p/334732028)

SimCLR

**image retrival**

[基于内容的图像检索技术：从特征到检索](https://zhuanlan.zhihu.com/p/46735159)

**3.7  NLP、OCR**

**NLP:**

[NLP中的Attention原理和源码解析](https://zhuanlan.zhihu.com/p/43493999)

[NLP分词算法深度综述](https://zhuanlan.zhihu.com/p/50444885)

Transformer：

[【NLP】Transformer模型原理详解](https://zhuanlan.zhihu.com/p/44121378)

[谷歌大改Transformer注意力，速度、内存利用率都提上去了](https://zhuanlan.zhihu.com/p/269751265)

Bert

[【NLP】Google BERT模型原理详解](https://zhuanlan.zhihu.com/p/46652512)

[FastBERT：又快又稳的推理提速方法](https://zhuanlan.zhihu.com/p/127869267)

GPT


**OCR:**

textCNN:

https://zhuanlan.zhihu.com/p/77634533

CTC：

[一文读懂CRNN+CTC文字识别](https://zhuanlan.zhihu.com/p/43534801)

CRNN：

[理解文本识别网络CRNN](https://zhuanlan.zhihu.com/p/71506131)










