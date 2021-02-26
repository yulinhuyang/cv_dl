## 分割

### 语义分割

#### FCN


##### 网络结构 


**原理**

全卷积网络，进行图像进行像素级的分类，CNN最后的全连接层换成卷积层，输出的是一张已经Label好的图



FCN-8s：

下采样：多个conv和+一个max pooling 叠加

上采样：首先进行pool4+2x upsampled feature逐点相加，然后又进行pool3+2x upsampled逐点相加，即进行更多次特征融合

对比：FCN-32s < FCN-16s < FCN-8s，即使用多层feature融合有利于提高分割准确性

**上采样方式** 

Resize，如双线性插值直接缩放，类似于图像缩放（这种方法在原文中提到）

Deconvolution，也叫Transposed Convolution

upsampling的意义：将小尺寸的高维度feature map恢复回去，以便做pixelwise prediction，获得每个点的分类信息

#### U-Net


##### 网络结构

