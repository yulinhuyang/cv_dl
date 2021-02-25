
### 开源代码






### 模型


**GAN**

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

