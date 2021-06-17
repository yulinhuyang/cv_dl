
参考：

https://github.com/WZMIAOMIAO/deep-learning-for-image-processing

https://space.bilibili.com/18161609

https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/yolov3_spp

## yolo系列

### 1 yolo V1

1 网络划分SXS个网络，object中心落在网格，则网格负责预测这个object

2 每个网络预测B个bounding box，每个bounding box包含预测位置+ confidence ，还有C个类别的分数。x,y是相对grid cell的， wh相对于整个图像的，conf:预测目标和真实目标的conf

3 网络结构：flatten fc -> fc reshape

4 损失函数（3 loss b c c S ）： bounding box + confidence损失+ classes损失,sum-squared error :误差平方和
		w h 开根号，平衡大小目标损失计算
		confidence损失: obj+ no obj
		
5  问题:群体小目标问题；目标比例问题

### 2 yolo V2


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
		
		


### 3 yolov3 整合

1  backbone修改

   换成Daknet19 -->Darknet53
	
   darknet和resnet对比：没有max-pooling层，通过stride=2的conv实现pooling
   
   CBL：使用了BN,没有bias参数

2  三个预测特征层（13*13、26*26、52*52）预测，每个层使用3种尺度，每个尺度3个box

   bounding box priors:
   
   COCO 输出大小：N * N *[3*(4 + 1 + 80)],4个坐标，1个conf,80个class score
   
   则N分别为13、26、52
   
   深度方向进行拼接

            feat26
            feat26
            feat13
             | 
  outfeat13 <-—— concat feat13        预测大目标
             |
  outfeat26 <--- concat feat26        预测中等目标
             |
  outfeat52 <--- concat feat52        预测小目标
  
   
3 目标边界框预测
  
  从cell左上角开始预测，预设值：cx cy pw ph,预测值：tx ty tw  th
  
  sigmoid将tx  ty进行映射
  
  算出来的bx by  bw  bh
  
4  正负样本匹配

   使用logistic regression预测

5  损失计算：lambda1*置信度损失Lconf+ lambda2*分类损失Lcls+ lambda3*定位损失
   
   置信度损失：sigmoid（C）--> BCE(二值交叉熵损失)
   
	   Oi:预测目标边界框与真实目标边界框的IOU 
	   
	   ci通过sigmoid函数得到的预测置信度,N正负样本格式
	
   类别损失：二值交叉熵损失，经过sigmoid处理sigmoid(Cij)  
   
   定位损失：误差平方和计算L2损失

   
### 4 yoloV3 SPP  yoloV3 SPP-ultralytics

**1  mosaic增强**
	
	多张图像拼接在一起，进行预测
	
	增加了数据多样性，目标个数，BN能一次性统计多张图像的参数，相当于增大了BN的大小

**2  SPP结构**
	
	和SPP-net的SPP不一样
	
	darknet53输出之后替换conv set为: SPP结构 + CONV1

	三个分支：maxpool 5X5/1  maxpool 9X9/1  maxpool 13X13/1
	       |
	concat融合 

**3  CIOU loss**
    
	IOU -->GIOU(差集/外接框面积比) -->DIOU（中心点距离） -->CIOU(长宽比)
	
	IOU loss = 1 - IOU
	
    GIOU退化：水平或者垂直相交
	
	DIOU loss；能直接最小化两个boxes之间的距离（d*d - C*C）
	
**4  Focal loss**
   
   yolov3作者尝试，map有下降
   
   class imbalance问题
   
   正负样本loss平衡：CE(pt) = -alpha t*log(pt)
   
   难易样本权重平衡：FL(pt) = -（1-pt）^gamma *log(pt)
   
   合一：gamma=2.0,alpha=0.25
   
   FL(p) = -alpha*(1-p)^gamma*log(p)   if y = 1
		   -(1-alpha)*p^gamma*log(1-p)  otherwise
   
   Focal loss 尽可能标注正确
   		
### 5   yolov3 SPP 源码

**训练代码**

	混合精度训练

	cfg中filters = [5+n]*3   class=n

	VOC转yolo格式 

	accumulate ：迭代多个batch更新一次参数

	freeze:只训练后面三个预测器参数

	OPENCV 读入图片是BGR格式，PIL是RGB格式

	最大边长，等比例缩放：letterbox
	   
		img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
		# Convert
		img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
		img = np.ascontiguousarray(img)

		img = torch.from_numpy(img).to(device).float()
      

**解析cfg文件**

类型说明
	
		[convolutional]
		batch_normalize=1   是否使用batch_normalize（使用bn,卷积层bias设置false）
		filters=32          卷积核个数
		size=3              卷积核尺寸
		stride=1            步长
		pad=1               是否启用padding
		
		[shortcut]
		from            与前面哪一层的输出进行融合
		activation      线性激活
		
		[maxpool]
		stride = 1  
		size = 5
		
		[route]
		layers= -2    返回到某一层的输出
		
		[route]
		layers=-1,-3,-5    拼接多层
		
		[yolo]层               预测器之后的层
		mask = 0,1,2          
		anchors = 10,13,16,30....	  小目标anchor + 中目标anchor + 大目标anchor
		class= 80               目标类别数
	
**模型代码**

isnumeric  判断是否是numeric类型
	

predictor：的conv没有Bn层,激活函数是linear,其他的都是leaky激活
	
model.py： anchor_vec 将anchors大小缩放到grid尺度（feature map）
	
	# 将anchors大小缩放到grid尺度
	self.anchor_vec = self.anchors / self.stride
	# batch_size, na, grid_h, grid_w, wh,
	# 值为1的维度对应的值不是固定值，后续操作可根据broadcast广播机制自动扩充
	self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

	# build xy offsets 构建每个cell处的anchor的xy偏移量(在feature map上的)
	if not self.training:  # 训练模式不需要回归到最终预测boxes,只需要计算损失
    
	p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

执行transpose、permute之后，内存不再连续，需要再执行contiguous。torch.view等方法操作需要连续的Tensor。
	
	io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy 计算在feature map上的xy坐标
	io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method 计算在feature map上的wh
    io[..., :4] *= self.stride  # 换算映射回原图尺度
	
	# yolo_out收集每个yolo_layer层的输出
    # out收集每个模块的输出
    yolo_out, out = [], []
