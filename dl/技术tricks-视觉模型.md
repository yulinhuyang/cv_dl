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

https://zhuanlan.zhihu.com/p/54289848

https://zhuanlan.zhihu.com/p/133805433

Resnext:    https://zhuanlan.zhihu.com/p/51075096

Res2net

ResNeSt:    https://zhuanlan.zhihu.com/p/132655457

Inception:   https://zhuanlan.zhihu.com/p/37505777

MobileNet:   

https://zhuanlan.zhihu.com/p/70703846

https://zhuanlan.zhihu.com/p/80177088

Senet：  https://zhuanlan.zhihu.com/p/65459972

SqueezeNet：  https://zhuanlan.zhihu.com/p/31558773  

ShuffleNet:  https://zhuanlan.zhihu.com/p/67009992

efficientnet： https://zhuanlan.zhihu.com/p/96773680

hrnet:  https://zhuanlan.zhihu.com/p/134253318


**3.2 检测：**

**Two-stage**

Faster RCNN:  

https://zhuanlan.zhihu.com/p/31426458

https://zhuanlan.zhihu.com/p/145842317

Mask RCNN:

https://zhuanlan.zhihu.com/p/37998710

Mask score RCNN:  https://zhuanlan.zhihu.com/p/111722103

**One-stage**

SSD： https://zhuanlan.zhihu.com/p/33544892

Yolo

https://zhuanlan.zhihu.com/p/37850811

https://zhuanlan.zhihu.com/p/76802514

https://zhuanlan.zhihu.com/p/74540100

gaussion yolo

RetinaNet：  https://zhuanlan.zhihu.com/p/133317452

**Anchor-Free**

https://zhuanlan.zhihu.com/p/62103812

FCOS：  https://zhuanlan.zhihu.com/p/62869137

NAS_FPN：  https://zhuanlan.zhihu.com/p/63300940

*FPN专题*	 ： https://zhuanlan.zhihu.com/p/148738276

**mmdetetion补充**
**RCNN改进**

Cascade R-CNN：  https://zhuanlan.zhihu.com/p/42553957

Cascade Mask R-CNN：  

Foveabox 

Hybrid Task Cascade：  https://zhuanlan.zhihu.com/p/57629509

Guided Anchoring

FreeAnchor

ATSS: https://zhuanlan.zhihu.com/p/115407465

FSAF： https://zhuanlan.zhihu.com/p/58782838

Foveabox： https://zhuanlan.zhihu.com/p/63190983

PointRend： https://zhuanlan.zhihu.com/p/98351269


**不均衡问题**

OHEM: https://zhuanlan.zhihu.com/p/78837273	

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

https://zhuanlan.zhihu.com/p/269751265

GCNet	

InstaBoost

GRoIE	

DetectoRS

Detr 	

**3.3 跟踪**

SiamFC

SiamRPN	

SiamRPN++

SiamMask

SiamRCNN

SiamFC++

DIMP	

FairMOT	

CenterTrack		

**3.4 分割**

**语义分割**

FCN 

Unet	

Segnet	

Pspnet	

Deeplabv3

Deeplabv3plus

PSANet 	

UPerNet	

NonLocal Net：   

https://zhuanlan.zhihu.com/p/64988633

https://zhuanlan.zhihu.com/p/77834369

EncNet	

CCNet 

DANet 

GCNet

ANN

OCRNet	

**实例分割**

FCIS

YOLACT-700	

YOLACT++

PolarMask

SOLO	

SOLOV2	

PointRender

BlendMask

TensorMask	

conv instance	


**3.5 GAN**

Pix2Pix(Pix2PixHD)

CycleGAN(StarGAN)

SRGAN(ESRGAN)

Progressing GAN

StyleGAN

**3.6 人脸人体、  3D、Reid、度量、image retrival**

MTCNN		
		构建图像金字塔
		P-Net 、R-Net、o-Net
		Soft-NMS:降低置信度
		5个关键点
RetinaFace	
		SSH检测模块
		关键点回归Dense Regression Branch：2D 3D映射图卷积

**人脸loss**		

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

3DMM、Firstorder、Prnet、3DFFA

**3D人体**

SMPL、SMPL+D、PIFUHD

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

MOCO

SimCLR

**image retrival**		

**3.7  NLP、OCR**

**NLP:**

Transformer

Bert

GPT

**OCR:**

textCNN

CTC

CRNN










