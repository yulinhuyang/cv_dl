## 分割

### Two-stage


#### Faster RCNN:


##### 网络结构 


<div align="center"> <img src="fastrcnn1.jpg" width="70%"/> </div><br>

<div align="center"> <img src="fastrcnn2.jpg" width="50%"/> </div><br>

<div align="center"> <img src="fastrcnn3.png" width="70%"/> </div><br>

**RPN**

RPN网络实际分为2条线，上面一条通过softmax分类anchors获得positive和negative分类，下面一条用于计算对于anchors的bounding box regression偏移量，以获得精确的proposal。而最后的Proposal层则负责综合positive anchors和对应bounding box regression偏移量获取proposals，同时剔除太小和超出边界的proposals。

其实整个网络到了Proposal Layer这里，就完成了相当于目标定位的功能。

其实RPN最终就是在原图尺度上，设置了密密麻麻的候选Anchor。然后用cnn去判断哪些Anchor是里面有目标的positive anchor，哪些是没目标的negative anchor。所以，仅仅是个二分类而已！

那么Anchor一共有多少个？原图800x600，VGG下采样16倍，feature map每个点设置9个Anchor，所以：
800/16 X 600/16 X9 = 50X38X9 = 17100

RPN网络总结：

生成anchors -> softmax分类器提取positvie anchors -> bbox reg回归positive anchors -> Proposal Layer生成proposals


**proposal**

<div align="center"> <img src="fasterrcnn4.png" width="70%"/> </div><br>

<div align="center"> <img src="fasterrcnn5.png" width="70%"/> </div><br>

<div align="center"> <img src="fasterrcnn6.png" width="70%"/> </div><br>


**RoI Pooling**

<div align="center"> <img src="fasterrcnn7.png" width="50%"/> </div><br>

<div align="center"> <img src="fasterrcnn8.png" width="70%"/> </div><br>


#### Mask RCNN:


##### 网络结构 

<div align="center"> <img src="maskrcnn1.png" width="70%"/> </div><br>

<center class="half">
     <img src="maskrcnn2.png" width="60%"/><img src="maskrcnn3.png" width="20%"/>
</center>

<center class="half">
     <img src="maskrcnn4.png" width="50%"/><img src="maskrcnn5.png" width="30%"/>
</center>



maskrcnn  = ResNet-FPN+Fast RCNN+Mask

1  骨干网络ResNet-FPN，用于特征提取，另外，ResNet还可以是：ResNet-50,ResNeXt-50

2  头部网络，包括边界框识别（分类和回归）+mask预测

双线性插值: 本质上就是在x y两个方向上做线性插值。

**ROI ALign**

虚线部分表示feature map，实线表示ROI，这里将ROI切分成2x2的单元格。如果采样点数是4，那我们首先将每个单元格子均分成四个小方格（如红色线所示），每个小方格中心就是采样点。这些采样点的坐标通常是浮点数，所以需要对采样点像素进行双线性插值，就可以得到该像素点的值了。然后对每个单元格内的四个采样点进行maxpooling，就可以得到最终的ROIAlign的结果。

**LOSS计算**

需要注意的是，计算loss的时候，并不是每个类别的sigmoid输出都计算二值交叉熵损失，而是该像素属于哪个类，哪个类的sigmoid输出才要计算损失(如图红色方形所示)。并且在测试的时候，我们是通过分类分支预测的类别来选择相应的mask预测。这样，mask预测和分类预测就彻底解耦了。


#### mask scoring rcnn


##### 网络结构 

<div align="center"> <img src="maskscorercnn1.png" width="70%"/> </div><br>


MS RCNN和Mask Rcnn类似，只是在Mask Rcnn的基础上，增加了MaskIOU分支，整个网络是端到端的。

在选择ROI时，如果按照每个ROI的score来排序筛选ROI，会出现一个问题就是，置信度高的ROI并不一定BBOX的位置就准，后来作者尝试了使用IoU来筛选ROI，发现效果要好。


#### ssd


##### 网络结构 

<div align="center"> <img src="ssd1.png" width="70%"/> </div><br>

<div align="center"> <img src="ssd2.png" width="70%"/> </div><br>

<div align="center"> <img src="ssd3.png" width="30%"/> </div><br>


**主要特点**

从YOLO中继承了将detection转化为regression的思路，一次完成目标定位与分类

基于Faster RCNN中的Anchor，提出了相似的Prior box；

加入基于特征金字塔（Pyramidal Feature Hierarchy）的检测方式，即在不同感受野的feature map上预测目标

SSD采用金字塔结构，即利用了conv4-3/conv-7/conv6-2/conv7-2/conv8_2/conv9_2这些大小不同的feature maps，在多个feature maps上同时进行softmax分类和位置回归


**生成prior box**

SSD按照如下规则：

以feature map上每个点的中点为中心，生成一些列同心的prior box

正方形prior box最小边长为和最大边长为

每在prototxt设置一个aspect ratio，会生成2个长方形，长宽为

SSD使用感受野小的feature map检测小目标，使用感受野大的feature map检测更大目标。


#### yolov3


##### 网络结构 

<div align="center"> <img src="yolov31.png" width="70%"/> </div><br>

<center class="half">
     <img src="yolov32.png" width="60%"/><img src="yolov33.png" width="30%"/>
</center>

<center class="half">
     <img src="yolov34.png" width="40%"/><img src="yolov35.png" width="50%"/>
</center>


**YOLO**

将输入图像分成SxS个格子，若某个物体 Ground truth 的中心位置的坐标落入到某个格子，那么这个格子就负责检测出这个物体。

**darknet-53**

借用了resnet的思想，在网络中加入了残差模块，这样有利于解决深层次网络的梯度问题，每个残差模块由两个卷积层和一个shortcut connections.1,2,8,8,4代表有几个重复的残差模块，整个v3结构里面，没有池化层和全连接层，网络的下采样是通过设置卷积的stride为2来达到的，每当通过这个卷积层之后图像的尺寸就会减小到一半。
对于多尺度检测来说，采用多个尺度进行预测，具体形式是在网络预测的最后某些层进行上采样拼接的操作来达到。

对于这三种检测的结果并不是同样的东西，这里的粗略理解是不同给的尺度检测不同大小的物体。网络的最终输出有3个尺度分别为1/32，1/16，1/8。

**concat**

张量拼接。将darknet中间层和后面的某一层的上采样进行拼接。拼接的操作和残差层add的操作是不一样的，拼接会扩充张量的维度，而add只是直接相加不会导致张量维度的改变。

**K-means**

聚类得到先验框的尺寸，为每种下采样尺度设定3种先验框，总共聚类出9种尺寸的先验框。
yolo v3对bbox进行预测的时候，采用了logistic regression。yolo v3每次对b-box进行predict时，输出(tx,ty,tw,th,to)​ ,然后通过公式1计算出绝对的(x, y, w, h, c)。

**logistic**

预测对象类别时不使用softmax，改成使用logistic的输出进行预测。这样能够支持多标签对象。


#### gaussion yolo


##### 网络结构 

<div align="center"> <img src="gyolo1.png" width="70%"/> </div><br>

<center class="half">
     <img src="gyolo2.png" width="50%"/><img src="gyolo3.png" width="30%"/>
</center>


yolo网络仅是预测了前景执行度与类别置信度，但不知道框的准确度。思路与GIOU-Net差不多。需要预测一个框的准确度。

output：在原来Yolov3的基础上，每个anchor多预测四个值，表示x,y,w,h的不确定度。box loss为有原来的平方Loss,改成了下边的，其他所有的都一样。预测的时添加了一个不确定性.


#### yolov4


##### 网络结构 

<div align="center"> <img src="yolov41.png" width="80%"/> </div><br>

<center class="half">
     <img src="yolov42.png" width="40%"/><img src="yolov43.png" width="40%"/>
</center>

<div align="center"> <img src="yolov45.png" width="50%"/> </div><br>


**Bag of freebies**

在目标检测中是指：用一些比较有用的训练技巧来训练模型，从而使得模型取得更好的准确率但是不增加模型的复杂度。
几何增强以及色彩增强：

解决目标遮挡及不足：random erase和CutOut 、DropOut，DropConnect， DropBlock

MIX-UP、style-transfer GAN

数据不均衡问题：Focal-loss、label-smooth

LOSS :  在选择ROI时，如果按照每个ROI的score来排序筛选ROI，会出现一个问题就是，置信度高的ROI并不一定BBOX的位置就准，后来作者尝试了使用IoU来筛选ROI，发现效果要好。

**Bag of specials**

增大感受域（ASFF，ASPP，RFB这些模块）

引入注意力机制（有两种一种是spatial attention，另外一种是channel attention，Channel+Spatial）

增加特征集成能力（Skip connection、FPN，ASFF，BiFPN)

注：Skip connection：用在encoder-decoder中比较多，最经典的Unet，融入了low-level和high-level的信息，在目标检测中主要用在了类似于pose这一类的检测任务中，例如DLA，Hourglass，以及最近的CenterNet

后处理：NMS，soft NMS，DIoU NMS


**采用**

Mosaic（马赛克）

数据增强，把四张图拼成一张图来训练，变相的等价于增大了mini-batch。这是从CutMix混合两张图的基础上改进。

Self-Adversarial Training(自对抗训练)

这是在一张图上，让神经网络反向更新图像，对图像做改变扰动，然后在这个图像上训练。这个方法，是图像风格化的主要方法，让网络反向更新图像来风格化图像。
CMBN

跨最小批的归一化（Cross mini-batch Normal）

修改的SAM：从SAM的逐空间的attention，到逐点的attention 

修改的PAN：把通道从相加（add）改变为concat，改变很小 


#### retinanet


##### 网络结构 


<div align="center"> <img src="retina1.png" width="80%"/> </div><br>

<div align="center"> <img src="retina2.png" width="40%"/> </div><br>

<div align="center"> <img src="retina3.png" width="70%"/> </div><br>

<div align="center"> <img src="retina4.png" width="20%"/> </div><br>


基础网络使用的是Resnet，在不同尺度的feature map建立金字塔，也就是FPN网络，这样就获得了丰富且多尺度的卷积特征金字塔，并且在FPN的每个level连接一个subnet用于回归和分类预测，这些subnet的参数是共享的，它相当于一个小型的FCN结构

正负样本不均衡、难易样本不均衡


#### FCOS


##### 网络结构 


<div align="center"> <img src="fcos1.png" width="70%"/> </div><br>

<div align="center"> <img src="fcos2.png" width="60%"/> </div><br>

<div align="center"> <img src="fcos3.png" width="70%"/> </div><br>

<div align="center"> <img src="fcos4.png" width="60%"/> </div><br>

<div align="center"> <img src="fcos5.png" width="70%"/> </div><br>

<div align="center"> <img src="fcos6.png" width="60%"/> </div><br>


FCOS:

检测器实现了proposal free和anchor free，显著的减少了设计参数的数目。设计参数通常需要启发式调整，并且设计许多技巧。

FCOS可以作为二阶检测器的区域建议网络(RPN)，其性能明显优于基于锚点的RPN算法。

在FCOS中，如果位置 (x,y) 落入任何真实边框，就认为它是一个正样本，它的类别标记为这个真实边框的类别。
这样会带来一个问题，如果标注的真实边框重叠，位置 (x,y) 映射到原图中落到多个真实边框，这个位置被认为是模糊样本.

center-ness，中心度取值为0,1之间，使用交叉熵损失进行训练。并把损失加入前面提到的损失函数中，中心度可以降低远离对象中心的边界框的权重。因此，这些低质量边界框很可能被最终的非最大抑制（NMS）过程滤除，从而显着提高了检测性能。


#### NASFPN


##### 网络结构 

<div align="center"> <img src="nas1.png" width="60%"/> </div><br>

<div align="center"> <img src="nas2.png" width="70%"/> </div><br>

<center class="half">
     <img src="nas4.png" width="50%"/><img src="nas3.png" width="30%"/>
</center>


NAS 利用强化学习训练控制器在给定的搜索空间中选择最优的模型架构。

NAS-FPN 采用 RNN 作为控制器，使用该控制器来产生一串信息，用于构建不同的连接。

具体来说，有以下6个步骤：

Step1: 从节点集合中选取第一个特征图节点，H1， 作为融合输入
Step2: 从节点集合中选取第二个特征图节点，H2，作为另一个融合输入
Step3: 从节点集合中选取第三个特征图节点，H3，作为输出分辨率
Step4: 从操作池选择融合操作

Step4，sum和global pooling，类似注意力机制。输入的特征层使用最近邻采样或者max pooling来调整到输出分辨率，merged特征层总会跟着ReLu, 3x3卷积和一个BN层。

Step5: 新融合的特征图作为新的节点，再放进节点集合里
Step6: 遍历以上步骤，直到填满输出金字塔的每一层


#### FPN专题


##### 网络结构 

<div align="center"> <img src="fpn1.png" width="50%"/> </div><br>

<div align="center"> <img src="fpn2.png" width="50%"/> </div><br>

**自上而下单向融合:**

<div align="center"> <img src="fpn3.png" width="50%"/> </div><br>


**简单双向融合:**

<div align="center"> <img src="fpn4.png" width="40%"/> </div><br>

复杂的双向融合：ASFF、NAS-FPN和BiFPN

<center class="half">
     <img src="fpn5.png" width="30%"/><img src="fpn6.png" width="30%"/><img src="fpn7.png" width="30%"/>
</center>


ASFF：（论文：Learning Spatial Fusion for Single-Shot Object Detection）作者在YOLOV3的FPN的基础上，研究了每一个stage再次融合三个stage特征的效果。如下图。其中不同stage特征的融合，采用了注意力机制，这样就可以控制其他stage对本stage特征的贡献度。

NAS-FPN和BiFPN，都是google出品，思路也一脉相承，都是在FPN中寻找一个有效的block，然后重复叠加，这样就可以弹性的控制FPN的大小。



**递归FPN**

<center class="half">
     <img src="fpn8.png" width="40%"/><img src="fpn9.png" width="60%"/>
</center>

DetectoRS，效果之好令人惊讶，使用递归FPN的DetectoRS是目前物体检测（COCO mAP 54.7）、实体分割和全景分割的SOTA，太强悍了。


