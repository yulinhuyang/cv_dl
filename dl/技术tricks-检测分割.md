### 开源代码:

**检测：**

open-mmlab/mmdetection
 
https://github.com/hoya012/deep_learning_object_detection
 
**分割：**

开源库：

open-mmlab/mmsegmentation

https://github.com/qubvel/segmentation_models.pytorch
 

### tricks


**目标检测比赛tricks**

roger：目标检测中的Tricks：https://zhuanlan.zhihu.com/p/138855612
 
初识CV：目标检测比赛中的tricks（已更新更多代码解析）： https://zhuanlan.zhihu.com/p/102817180
 
初识CV：目标检测比赛笔记： https://zhuanlan.zhihu.com/p/137567177
 
Slumbers：目标检测任务的优化策略tricks：https://zhuanlan.zhihu.com/p/56792817

 
Bag of Freebies for Training Object Detection Neural Networks

Amusi：亚马逊提出：目标检测训练秘籍（代码已开源）： https://zhuanlan.zhihu.com/p/56700862
 
目标检测比赛的奇技淫巧（tricks）_Snoopy_Dream-CSDN博客： https://blog.csdn.net/e01528/article/details/82354477



**加大batch_size方法：**

多卡、混合精度、MOCO（cross batch memory）

Todd：PyTorch Parallel Training（单机多卡并行、混合精度、同步BN训练指南文档）

论文总结： 
[图像语义分割(Semantic segmentation) Survey](https://zhuanlan.zhihu.com/p/36801104)

 
 ### 模型总结
 
 #### 检测
 
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

#### 跟踪 


SiamFC：  https://zhuanlan.zhihu.com/p/66757733

SiamRPN	

SiamRPN++： [视觉目标跟踪之SiamRPN++](https://zhuanlan.zhihu.com/p/56254712)

SiamMask: [我对Siamese网络的一点思考（SiamMask）](https://zhuanlan.zhihu.com/p/58154634)

SiamRCNN

SiamFC++

DIMP	

FairMOT：  [MOT开源实时新SOTA | FairMOT](https://zhuanlan.zhihu.com/p/126558285)	

CenterTrack：  https://zhuanlan.zhihu.com/p/125395219		

## 分割 

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

 
