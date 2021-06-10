### 1 项目调研

####论文调研

**lihongdong 相关**

https://github.com/shiyujiao/cross_view_localization_CVFT

CVFT结构

[Optimal Feature Transport for Cross-View Image Geo-Localization ](https://github.com/kregmi/cross-view-image-matching)

[Where am I looking at? Joint Location and Orientation Estimation by Cross-View Matching](https://github.com/shiyujiao/cross_view_localization_DSM)

DSM结构

LRN结构（zhengzhedong）:Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization


https://github.com/layumi/University1652-Baseline

CVUSA/CVACT： https://github.com/Liumouliu/OriCNN


netvlad:

[CVM-Net_Cross-View_Matching](https://github.com/david-husx/crossview_localisation)

Bridging the Domain Gap for Ground-to-Aerial Image Matching


**分割材料**

Large Kernel Matters GCN:

https://github.com/ZijunDeng/pytorch-semantic-segmentation

建筑物分割：

https://github.com/milesial/Pytorch-UNet

分割识别：

Image-Based Camera Localization by Leveraging Semantic Information in Urban Environments

Semantically-Aware Attentive Neural Embeddings for Long-Term 2D Visual Localization 

Semantic Match Consistency for Long-Term Visual Localization

**image-retrival论文**

Fine-tuning CNN Image Retrieval with No Human Annotation

[图像检索综述](https://zhuanlan.zhihu.com/p/46735159)

特征处理方式：VLAD、Fisher Vector

快速查找技术:LSH    Hamming Embedding IMI     NO-IMI       PQ 

查找优化-Hamming Embedding

SOLAR: Second-Order Loss and Attention for Image Retrieval

**REID相关**

Beyond Part Models: Person Retrieval with Refined Part Pooling

PCB：Part-based Convolutional Baseline：part-level的特征进行行人重识别提供了细粒度

Refined Part Pooling：精炼池化，重新分配这些极端值到它相似的片段

[Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://tjjtjjtjj.github.io/2019/05/17/BNNeck/)

#### 开源工程

image-retrival 相关 

https://github.com/almazan/deep-image-retrieval

https://github.com/PyRetri/PyRetri


reid库

https://github.com/JDAI-CV/fast-reid

https://github.com/michuanhaohao/reid-strong-baseline

Metric-learning:

https://github.com/KevinMusgrave/pytorch-metric-learning


图像增强库：

https://github.com/albumentations-team/albumentations



### 2 方法尝试

#### 评价指标尝试

REID 中的CMC和MAP

https://wrong.wang/blog/20190223-reid%E4%BB%BB%E5%8A%A1%E4%B8%AD%E7%9A%84cmc%E5%92%8Cmap/

Triplet loss + Contrastive Loss


#### 模型尝试

[训练trick](https://zhuanlan.zhihu.com/p/93806755)

[CNN训练分类任务的优化策略(tricks)](https://zhuanlan.zhihu.com/p/53849733)

[深度学习 cnn trick合集](https://zhuanlan.zhihu.com/p/137940586)

[cnn结构设计技巧-兼顾速度精度与工程实现](https://zhuanlan.zhihu.com/p/100609339)

训练策略：warm-up、WarmupMultiSetpLR、WarmupCosineLR

优化器：Stochastic Weight Averaging

数据增强：

Random erasing( )  RandomPatch( )  Auto Mix   Color-jitter  亮度对比度

GAN增强图像：夜景白天问题，纹理修复问题。


Weight Standard： https://github.com/joe-siyuan-qiao/WeightStandardization

Space2depth: https://blog.csdn.net/shwan_ma/article/details/103604372

一欧元滤波：https://cristal.univ-lille.fr/~casiez/1euro/

二维图像三维增强：

https://www.learnopencv.com/rotation-matrix-to-euler-angles/

特殊点旋转：

https://stackoverflow.com/questions/17087446/how-to-calculate-perspective-transform-for-opencv-from-rotation-angles

全局点： 2 维转3维，再转2维，求解变换矩阵

**backbone**

IBN(跨域):https://github.com/XingangPan/IBN-Net

BNeck:

Pooling：RMAC、Gempooling（广义平均池化）

白化和降维：最后有带bias的全连接一层

大核: GCN结构、空洞卷积HDC结构设计

SE-block: https://github.com/hujie-frank/SENet

CBAM: https://github.com/Jongchan/attention-module

Nonlocal:

LeakyReLU与RelU

BN层价值与dropout

**后处理**

Rerank、Query expansion


**loss**

Metric-learning: CrossBatchMemory 、SimCLR、MOCO

hard-mining triplet loss

**部署**

libtorch：

https://pytorch.org/cppdocs/frontend.html

https://github.com/onnx/models

tensorRT：

https://github.com/Syencil/tensorRT

https://zhuanlan.zhihu.com/p/146030899

### 3 成功经验

**图像处理**

random erase、CoarseDropout、EnhanceColor、EnhanceContrast

pading对齐操作：cv2.copyMakeBorder

照片进行：柱面投影，为了和全景图的一致

分割减通道+膨胀

**backbone**

大核短层：平衡定位精度与大感受野

pooling层之间，要充分感知feature


分辨率：很重要，尽量不要损失分辨率，为了保住分辨率，在使用下采样之前要保证在这一层上有足够的感受野，这个感受野是相对感受野，

是指这一个下采样层相对于上一个下采样层的感受野，把两个下采样之间看成一个子网络的话，这个子网络必须得有一定的感受野才能将空间信息编码到下面的网络去，

而具体需要多大的相对感受野，只能实验，一般说来，靠近输入层的层空间信息冗余度最高，所以越靠近输入层相对感受野应该越小。同时在靠近输入层的层，这里可以合成一个大卷积核来降低计算量，

因为在输入端，每一层卷积的计算量都非常大。另外相对感受野也必须缓慢变换。

模型小型化：Stride 


weight height channel都很重要：特征图都需要

感受野与featuremap计算：全景图的计算问题

归一化：torch.nn.functional.normalize(features, dim=(1,2), p=2)

**loss**

逐条的loss

hard-mining的TripletMarginLoss, negtive来源:随机采集、周围区域裁剪、随机区域颜色抖动


**抖动pitch roll**

x-y-z ; p y r  右手坐标系

找到yaw图像之后，进行轻微抖动，进行精细匹配

检索：滑窗卷积

训练：DDP，torch.distributed.init_process_group + torch.nn.parallel.DistributedDataParallel

打点：

summary_writer 记录日志和log、dump图



