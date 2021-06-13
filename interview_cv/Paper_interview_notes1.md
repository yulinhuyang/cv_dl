
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing

## yolo系列

### yolo V1

### yolo V2


BN: 正则化模型，替代dropout层

high resolution Classifier: 224 ->416、448

conv with anchor box:
	
	每个anchor(prior)去负责预测目标中心落在某个grid cell区域内的目标
	
	logistic函数，限制每个anchor负责预测目标中心落在某个grid cell区域内的目标
	
	划分grid shell + logistic（限制grid shell）
	
	anchor预设值： cx cy pw ph
	
	网络预测值：tx ty tw th

Dimension Clusters: 聚类anchor

Fine-Grained Features:

	pass through层：宽高各缩小一半，channel变成4倍，直接concat到后面

multi-scale training 多尺度训练：每10轮，下降输入分辨率32进行训练

	CBL：con2d + BN + leaky-relu,使用BN层时，卷积里面的bais不起作用

head设置：去掉最后一个卷积层，换成3个head层(conv 3*3 1024 + conv1)
		
		预测5个box: (5+20)X5

		
