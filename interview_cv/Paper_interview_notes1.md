
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing

## yolo系列

### yolo V1

1 网络划分SXS个网络，object中心落在网格，则网格负责预测这个object

2 每个网络预测B个bounding box，每个bounding box包含预测位置+ confidence ，还有C个类别的分数。x,y是相对grid cell的， wh相对于整个图像的，conf:预测目标和真实目标的conf

3 网络结构：flatten fc -> fc reshape

4 损失函数（3 loss b c c S ）： bounding box + confidence损失+ classes损失,sum-squared error :误差平方和
		w h 开根号，平衡大小目标损失计算
		confidence损失: obj+ no obj
		
5  问题:群体小目标问题；目标比例问题

### yolo V2


1 BN: 正则化模型，替代dropout层

2 high resolution Classifier: 224 ->416、448

3 conv with anchor box:
	
	每个anchor(prior)去负责预测目标中心落在某个grid cell区域内的目标
	
	logistic函数，限制每个anchor负责预测目标中心落在某个grid cell区域内的目标
	
	划分grid shell + logistic（限制grid shell）
	
	anchor预设值： cx cy pw ph
	
	网络预测值：tx ty tw th

4  Dimension Clusters: 聚类anchor

5  Fine-Grained Features:

	pass through层：宽高各缩小一半，channel变成4倍，直接concat到后面

6 multi-scale training 多尺度训练：每10轮，下降输入分辨率32进行训练

	CBL：con2d + BN + leaky-relu,使用BN层时，卷积里面的bais不起作用

7  head设置：去掉最后一个卷积层，换成3个head层(conv 3*3 1024 + conv1)
		
		预测5个box: (5+20)X5

		
