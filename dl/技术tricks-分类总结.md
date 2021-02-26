### 开源代码

**Reid 分类：**
 
FastReID：目前最强悍的目标重识别开源库！
https://github.com/JDAI-CV/fast-reid

**图像增强：**

https://github.com/albumentations-team/albumentations


https://cloud.tencent.com/developer/article/1585733


https://github.com/aleju/imgaug


https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py


fast-autoaugment: 
https://github.com/kakaobrain/fast-autoaugment

augmix: 
https://github.com/google-research/augmix

mixup/cutout: 
https://github.com/PistonY/torch-toolbox

model
pretrained-models.pytorch: 
https://github.com/Cadene/pretrained-models.pytorch

**metric learning**

pytorch-metric-learning: 
https://github.com/KevinMusgrave/pytorch-metric-learning

topk optimization: 
https://github.com/BG2CRW/top_k_optimization


**loss**

 Class-balanced-loss-pytorch: 
https://github.com/vandit15/Class-balanced-loss-pytorch


**framework**

pytorchimageclassification: 
https://github.com/hysts/pytorch_image_classification

**backbone:**

Resnest:
https://github.com/zhanghang1989/ResNeSt/tree/master/resnest/torch
 IBN：
https://github.com/XingangPan/IBN-Net/tree/master/ibnnet

Gempooling


### tricks

**比赛 tricks：**

Slumbers：CNN训练分类任务的优化策略(tricks)： https://zhuanlan.zhihu.com/p/53849733

Luke：Kaggle首战Top 2%, APTOS 2019复盘总结+机器学习竞赛通用流程归纳：https://zhuanlan.zhihu.com/p/81695773
 
Gary：本科生晋升GM记录 & kaggle比赛进阶技巧分享：https://zhuanlan.zhihu.com/p/93806755
 
 
**其他tricks**

sticky：深度学习 cnn trick合集： https://zhuanlan.zhihu.com/p/137940586
 
数据竞赛Tricks集锦：  www.jiqizhixin.com


Bag of Tricks for Image Classification with Convolution Neural Networks: 

https://zhuanlan.zhihu.com/p/66393448


[imgaug学习笔记](https://blog.csdn.net/u012897374/article/details/80142744)



## 模型汇总 

###  ResNet：

[ResNet及其变种的结构梳理、有效性分析与代码解读](https://zhuanlan.zhihu.com/p/54289848)

[关于ResNeSt的点滴疑惑](https://zhuanlan.zhihu.com/p/133805433)


构建恒等映射:解决网络退化问题

#### 网络结构：


输入部分、输出部分和中间卷积部分（中间卷积部分包括如图所示的Stage1到Stage4共计四个stage）

网络之间的不同主要在于中间卷积部分的block参数和个数存在差异

**残差块:**

basic-block：

Bottleneck：使用了1x1卷积的bottleneck将计算量简化为原有的5.9%


**常见改进：**

改进downsample部分，减少信息流失：下采样后移

ResNet部分组件的顺序进行了调整：如果将ReLU放在原先的位置，那么残差块输出永远是非负的，这制约了模型的表达能力。ReLU移入了残差块内部。


#### 设计特点：

如果特征地图大小减半，滤波器的数量加倍以保持每层的时间复杂度

每个stage通过步长为2的卷积层执行下采样，而却这个下采样只会在每一个stage的第一个卷积完成，有且仅有一次。

最大池化与平均池化： 网络以平均池化层和softmax的1000路全连接层结束，实际上工程上一般用自适应全局平均池化 (Adaptive Global Average Pooling)，好处：一是节省计算资源，二是防止模型过拟合，提升泛化能力。更保险的操作，就是最大池化和平均池化都做，然后把两个张量拼



###   Resnext

Resnext:   [ResNeXt详解](https://zhuanlan.zhihu.com/p/51075096)

ResNeXt：ResNet + Inception，split-transform-merge

#### 网络结构：



ResNeXt的分支的拓扑结构是相同的，Inception V4需要人工设计

ResNeXt是先进行1X1卷积然后执行单位加，Inception V4是先拼接再执行 1X1 卷积

#### 设计特点：

分组卷积： 介于普通卷积核深度可分离卷积的




###  Res2net


###  ResNeSt

ResNeSt:   [ResNeSt：Split-Attention Networks](https://zhuanlan.zhihu.com/p/132655457)

Split-Attention Networks，主要： Multi-path 和 Feature-map Attention思想

#### 网络结构：



Multi-path：GoogleNet、ResNeXt

SE-Net： 通过自适应地重新校准通道特征响应来引入通道注意力（channel-attention）

SK-Net： 通过两个网络分支引入特征图注意力（feature-map attention）






Inception:   https://zhuanlan.zhihu.com/p/37505777

MobileNet:   

[轻量级神经网络“巡礼”（二）—— MobileNet，从V1到V3](https://zhuanlan.zhihu.com/p/70703846)

[MobileNet 详解深度可分离卷积](https://zhuanlan.zhihu.com/p/80177088)

Senet：  https://zhuanlan.zhihu.com/p/65459972

SqueezeNet：  https://zhuanlan.zhihu.com/p/31558773  

ShuffleNet:  [轻量级神经网络“巡礼”（一）—— ShuffleNetV2](https://zhuanlan.zhihu.com/p/67009992)

efficientnet： [令人拍案叫绝的EfficientNet和EfficientDet](https://zhuanlan.zhihu.com/p/96773680)

hrnet:  [打通多个视觉任务的全能Backbone:HRNet](https://zhuanlan.zhihu.com/p/134253318)

