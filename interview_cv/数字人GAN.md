数字人GAN调研总结

### 论文调研

影随人动
	
	命令模式 + 检测 + reid判断 + openpose
	
	人体关键点PAF：https://github.com/CMU-Perceptual-Computing-Lab/openpose
	
	Focus loss 

光源估计
	
	Learning to Predict Indoor Illumination from a Single Image
	
	DeepLight: Learning Illumination for Unconstrained Mobile Mixed Reality 

人脸相关：
	
	https://github.com/biubug6/Face-Detector-1MB-with-landmark
	
	https://github.com/deepinsight/insightface
	
	loss演进：https://blog.csdn.net/weixin_43013761/article/details/100019718
	
	softmax —> Triplet ->Center(加类内距离) -> L - softmax(cos(mx)) -> A-softmax(SphereFace 限b=0,w=1)->AM-softmax(CosFace cos(theta)-m )-->ARCface(cos(theta+m))
	
	circle loss
	
	MTCNN、hourglass 
	
	PFLD
	
GAN:
		
	Conditional GAN --> Stack GAN --> Cycle GAN --> Star GAN 
	
	Wasserstein GAN ——> Wasserstein GAN GP
	
	Pix2Pix->Pix2PixHD
	
	SRGAN -> ESRGAN
	
	Growing GAN ->StyleGAN ->StyleGAN2
	
	http://www.gwylab.com/
	
	DeblurGAN，人脸超分(PSNR、SSIM)
	
3D人脸与效果：
	
	GANFIT

	face2face
	
	Pifu ->PifuHD
	
	SMPL 
	
	First Order Motion： https://github.com/AliaksandrSiarohin/first-order-model

	
	PRNet、3DDFA、3DMM
	
###开源项目

影随人动：举手判断逻辑(各个方向的角度)

###成功经验


caffe看参数：https://dgschwend.github.io/netscope/#/editor

caffe看网络结构：http://ethereon.github.io/netscope/#/editor

torch.nn.CrossEntropyLoss =  torch.nn.LogSoftmax + torch.nn.NLLLoss

PyTorch 里面的 torch.nn.CrossEntropyLoss (输入是我们前面讲的 logits，也就是 全连接直接出来的东西)。

这个 CrossEntropyLoss 其实就是等于 torch.nn.LogSoftmax + torch.nn.NLLLoss。

softmax把分类输出标准化成概率分布，cross-entropy（交叉熵）刻画预测分类和真实结果之间的相似度。
