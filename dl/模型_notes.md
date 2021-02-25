
###  ResNet：

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

ResNeXt：ResNet + Inception，split-transform-merge

#### 网络结构：



ResNeXt的分支的拓扑结构是相同的，Inception V4需要人工设计

ResNeXt是先进行1X1卷积然后执行单位加，Inception V4是先拼接再执行 1X1 卷积

#### 设计特点：

分组卷积： 介于普通卷积核深度可分离卷积的

###  ResNeSt

Split-Attention Networks，主要： Multi-path 和 Feature-map Attention思想

#### 网络结构：



Multi-path：GoogleNet、ResNeXt

SE-Net： 通过自适应地重新校准通道特征响应来引入通道注意力（channel-attention）

SK-Net： 通过两个网络分支引入特征图注意力（feature-map attention）

### Inception

Inception v1：多个filter size, 定位不同size的object

Inception v2:  5*5 filter换成了俩个3*3, 3*3 filter换成了1*3+3*1

Inception v4: 修改了Inception的Stem ,添加了Reduction block


#### 网络结构：



### MobileNet

深度可分离卷积: 深度卷积 + 逐点卷积

#### 网络结构：




