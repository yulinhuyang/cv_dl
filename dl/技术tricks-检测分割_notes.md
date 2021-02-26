## 分割

### 语义分割

#### FCN


##### 网络结构 


**原理**

全卷积网络，进行图像进行像素级的分类，CNN最后的全连接层换成卷积层，输出的是一张已经Label好的图

FCN-8s：

下采样：多个conv和+一个max pooling 作为模块，反复叠加

上采样：首先进行pool4+2x upsampled feature逐点相加，然后又进行pool3+2x upsampled 逐点相加 ，即进行更多次特征融合

对比：FCN-32s < FCN-16s < FCN-8s，即使用多层feature融合有利于提高分割准确性

**上采样方式** 

Resize，如双线性插值直接缩放，类似于图像缩放（这种方法在原文中提到）

Deconvolution，也叫Transposed Convolution

upsampling的意义：将小尺寸的高维度feature map恢复回去，以便做pixelwise prediction，获得每个点的分类信息

#### U-Net


##### 网络结构


**原理**

首先进行Conv+Pooling下采样；然后Deconv反卷积进行上采样，crop之前的低层feature map，进行融合；然后再次上采样。重复这个过程，直到获得输出388x388x2的feature map，最后经过softmax获得output segment map

**拼接**

U-Net采用将特征在channel维度拼接在一起，形成更“厚”的特征

特征融合：

FCN式的逐点相加，对应caffe的EltwiseLayer层，对应tensorflow的tf.add()

U-Net式的channel维度拼接融合，对应caffe的ConcatLayer层，对应tensorflow的tf.concat()

分割基本思路：

下采样+上采样：Convlution + Deconvlution／Resize

多尺度特征融合：特征逐点相加／特征channel维度拼接

获得像素级别的segement map：对每一个像素点进行判断类别

#### pspnet（Pyramid Scene Parsing Network）



##### 网络结构

**原理**

encoder: 使用了预训练的ResNet，里面使用了孔洞卷积（后面几层没有下采样，全部使用空洞卷积）。最后输出的feature map是原图的1/8

金字塔池化：使用金字塔池化模块，使用了四种尺寸的金字塔，池化所用的kerne分别1×1, 2×2, 3×3 and 6×6。池化之后上采样，然后将得到的feature map,包括池化之前的做一个级联（concatenate),后面接一个卷积层得到最终的预测图像

上采样使用双线性插值，训练使用了一个辅助的loss（中间层接出），是一个softmax loss 

#### SegNet
