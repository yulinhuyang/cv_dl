
expansion= 4

conv3_x-conv5_x 中残差结构，卷积层3的卷积个数是卷积1和2的四倍

	self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
						   kernel_size=1, stride=1, bias=False)  # unsqueeze channels
	self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
	

	#groups分组卷积
	self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
				   kernel_size=3, stride=stride, bias=False, padding=1)
	
	#四个块
	self.layer1 = self._make_layer(block, 64, blocks_num[0])
	self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
	self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
	self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
	
	def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

	
    #自适应平均池化下采样
	if self.include_top:
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
		self.fc = nn.Linear(512 * block.expansion, num_classes)	
				
	#1 不载入全连接层的方法,载入后修改
	missing_keys,unexpected_keys = net.load_state_dict(torch.load(model_weight_path),strict=False)
	
	# change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)
    net.to(device)
	
	#2 载入后修改字典，删掉全连接层参数
	torch.load(model_weight_path)
	

	
temperature parameter t控制softmax输出的平滑成的，t越大，结果越平滑。概率分布越平滑。

	
### mobilenet 

#### V1:

https://blog.csdn.net/qq_37541097/article/details/105771329

深度分离卷积

增强超参数 alpha beta


**Dw separable conv = DW + PW**

	DW conv:

		卷积核channel=1,

		输入特征矩阵channel=卷积核个数=输出特征矩阵channel

	PW conv: 普通的卷积，只是卷积核个数为1 

	alpha：卷积核个数倍率

	beta： 输入分辨率倍率
		
	DW卷积：部分的卷积核容易为0
	
#### V2：

Inverted residual block 倒残差结构

	传统残差(relu)： 1X1 conv降维 --> 3X3卷积 --->1x1 conv 升维

	倒残差（relu6）：1x1 升维--> 3x3卷积DW -->1X1卷积降维

	Relu激活函数对低维特征信息造成大量损失

	当时stride=1且输入特征矩阵与输出特征矩阵shape相同时才有shortcut连接


一个block由多个bottleneck组成

#### v3

更新block
	
	加入SE模块，更新激活函数，
	
	SE：          ->GP ->FC ->Relu ->FC ->hardsigmoid->
	 X__Residual |____________________________________|__scale --+ ---
	   |_______________________________________________________|
	 
	  两个环：G-F-R-F-S，残差内套SE
	  
	se_block -->_inverted_res_block
	
	
使用NAS搜索参数

重新设计耗时层结构
	
	减少第一个卷积层的卷积核个数，精简last stage
	
	Relu6激活函数和h-sigmoid激活函数
	
代码解析；

		def _make_divisible(ch, divisor=8, min_ch=None):  将输入的channel调整到离它最近的8的整数倍
		
			new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)

		#SE模块
		class SqueezeExcitation(nn.Module):
			def __init__(self, input_c: int, squeeze_factor: int = 4):
				super(SqueezeExcitation, self).__init__()
				squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
				self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
				self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

			def forward(self, x: Tensor) -> Tensor:
				scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
				scale = self.fc1(scale)
				scale = F.relu(scale, inplace=True)
				scale = self.fc2(scale)
				scale = F.hardsigmoid(scale, inplace=True)
				return scale * x
		
		
		InvertedResidualConfig  配置类
		inverted_residual_setting: List[InvertedResidualConfig]

		self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)
		layers: List[nn.Module] = []
	
		设定默认值
		bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
	

### ReNext

ResNeXt的精妙之处在于，该思路沿用到nlp里就有了 multi-head attention

拆分成32个path的 group conv,参数量下降为1/32

			   256-d in-------------
				  |                 |
			  256 1*1 128           |
				  |                 |
			  128 3*3 128 group=32  |
				  |                 |
			  128 1*1  256          |
				  |                 |
				  +------------------
				  |
			  256 out

block里面conv层数 > 3有意义

### shufflenet

**V1：**

channel shuffle

分组卷积-->每个组切分多块-->不同组编号相同的块放在一起，实现信息交流 

			stride =1 
						   |
				 ——————————
				|          |
			1X1  Gconv     |
				|          |
			channel shuffle|
				|          |
			3X3 DW conv    |
				|          |
			1x1 Gconv      |
				|          |
				|________add
						   |
						   
						   
			stride =2			   
							|
				 ——————————--
				|           |
			1X1  Gconv      |
				|           |
			channel shuffle |
				|           3x3 AVG pool stride =2
			3X3 DW conv s=2 |
				|           |
			1x1 Gconv       |
				|           |
				|________concat
							|
						   

不同组之间的信息交流

**V2:**

G1). 使用输入通道和输出通道相同的卷积操作；

G2). 谨慎使用分组卷积；

G3). 减少网络分支数；

G4). 减少element-wise(relu addTensor addBias)操作。

			stride =1
                           |
				    channel split
						   |
				 ——————————
				|          |
			1X1  conv      |
				|          |
			3X3 DW conv    |
				|          |
			1x1 conv       |
				|          |
				|________concat
						   |
						channel shuffle
						   |
						   
						   
			stride =2			   
							|
				 ——————————--
				|           |
			1X1  conv     3X3 DW conv s=2
				|           |
				|          1x1 conv
			3X3 DW conv s=2 |
				|           |
			1x1 conv        |
				|           |
				|________concat
							|
					channel shuffle 
					        |
					

### 详解Transformer中Self-Attention以及Multi-Head Attention

https://blog.csdn.net/qq_37541097/article/details/117691873




### pytorch 可视化

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
