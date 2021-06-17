### 1 resnet

https://blog.csdn.net/qq_37541097/article/details/104710784

超深的网络结构

提出residual结构

使用BN(代替dropout)

问题：

	梯度消失和爆炸（<1的数，传播越来越小，>1的数，传播越来越大）

	退化问题（深不如浅）
	
resnet在保持梯度相关性方面很优秀

恒等映射，H(x) = X

残差的思想都是去掉相同的主体部分，从而突出微小的变化。

当使用了残差网络时,就是加入了skip connection 结构,这时候由一个building block 的任务由: F(x) := H(x)，变成了F(x) := H(x)-x

F(x)学习变化更容易

	
	
**1 两种残差结构：**

浅层：

			64d
			  |——————
			  |      |
			3X3  64  |
			  |      |
			 relu    |
			  |      |
			3X3  64  |
			  |      |
			  +______|
			  |

深层：	  
	  
			256d ——————
			  |      |
			1X1  64  |
			  |      |
			relu     |
			  |      |
			3X3  64  |
			  |      |
			relu 64  |
			  |      |
			1X1  256 |
			  |      |
			  +______|
			  |
		  
	  
虚线的残差结构：

输入特征矩阵shape是[56,56,64],输出特征矩阵shape是[28,28,128]

			[56,56,64]
			  |——————————————————
			  |                 |
			3X3  128,stride =2  |
			  |                 |
			 relu              1x1 128 stride = 2
			  |                 |
			3X3  128            |
			  |                 |
			  +_________________|
			  |
			[28,28,128] 
			
			  relu
	

虚线深层残差结构：conv3 conv4 conv5 的第一层都有

		   56 X 56
			256d —————————— 
			  |            |
			1X1  128       |
			  |            |
			relu           |
			  |            |
		stride 2 3X3  128  1x1 512 stride=2   
			  |            |
			relu           |
			  |            |
			1X1  512       |
			  |            |
			  +____________| 
			  |
			28 28  512

主分支与shortcut的输出特征矩阵shape必须相同

这里是add到一起的

1X1的卷积核用来降维和升维

将特征矩阵的高和宽缩减为原来的一半，将深度channel调整成下一层残差结构所需要的channel

**2 BN层**

Batch Normalization的目的就是使我们的feature map满足均值为0，方差为1的分布规律

计算一批数据的每个通道的均值和方差（Sigma），计算出整个训练集的feature map然后在进行标准化处理

四大参数：gamma调整方差的大小，beta调整均值，可以学习参数的，是通过反向传播得到的。均值和方差是通过前向的时候一批批数据统计得到的。

注意问题：
	
	（1）训练时要将traning参数设置为True，在验证时将trainning参数设置为False。在pytorch中可通过创建模型的model.train()和model.eval()方法控制。

	（2）batch size尽可能设置大点，设置小后表现可能很糟糕，设置的越大求的均值和方差越接近整个训练集的均值和方差。

	（3）建议将bn层放在卷积层（Conv）和激活层（例如Relu）之间，且卷积层不要使用偏置bias，因为没有用。 
	

**3 网络结构**




	
###  2 mobilenet 

https://blog.csdn.net/qq_37541097/article/details/105771329







### 3 shufflenet

	





### 4  pytorch 可视化

analyze_feature_map.py

	analyze_weights_featuremap

	tensorboard
	
	获取特定层输出：
	
		outputs = []
		for name, module in self.features.named_children():
			x = module(x)
			if name in ["0", "3", "6"]:
				outputs.append(x)
	
	# [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.detach().numpy())
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])
	
	特征矩阵每一个channel所对应的是一个二维的特征矩阵，就像灰度图像一样，channel = 1
	
		plt.figure()
		for i in range(12):
			ax = plt.subplot(3, 4, i+1)
			# [H, W, C]
			plt.imshow(im[:, :, i], cmap='gray')
	
	gray：是否蓝绿展示
	
analyze_kernel_weight.py
	
	weights_keys = model.state_dict().keys()
	for key in weights_keys:
		# remove num_batches_tracked para(in bn)
		if "num_batches_tracked" in key:
			continue
		# [kernel_number, kernel_channel, kernel_height, kernel_width]
		weight_t = model.state_dict()[key].numpy()

		# calculate mean, std, min, max
		weight_mean = weight_t.mean()
		weight_std = weight_t.std(ddof=1)
		

### 5 sort 算法




### 6  PFLD
