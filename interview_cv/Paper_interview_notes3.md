
参考：

B站：Bubbliiiing、霹雳吧啦Wz、江大白

霹雳吧啦Wz:

https://github.com/WZMIAOMIAO/deep-learning-for-image-processing

https://space.bilibili.com/18161609

https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/yolov3_spp

[YOLO v3网络结构分析](https://blog.csdn.net/qq_37541097/article/details/81214953)

霹雳吧啦Wz:

https://github.com/bubbliiiing/yolov4-pytorch

[睿智的目标检测30——Pytorch搭建YoloV4目标检测平台](https://blog.csdn.net/weixin_44791964/article/details/106214657)

[睿智的目标检测26——Pytorch搭建yolo3目标检测平台](https://blog.csdn.net/weixin_44791964/article/details/105310627)

江大白

[深入浅出Yolo系列之Yolov3&Yolov4&Yolov5核心基础知识完整讲解](https://zhuanlan.zhihu.com/p/143747206)

[深入浅出Yolo系列之Yolov5核心基础知识完整讲解](https://zhuanlan.zhihu.com/p/172121380)

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


**自定义数据集：**


	# 检查每张图片后缀格式是否在支持的列表中，保存支持的图像路径
	# img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
	self.img_files = [x for x in f if os.path.splitext(x)[-1].lower() in img_formats]

	os.path.splitext(x)[-1] --->取后缀

        # 将数据划分到一个个batch中
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)
        # 记录数据集划分后的总batch数
        nb = bi[-1] + 1  # number of batches
		
	# 注意: 开启rect后，mosaic就默认关闭
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)

		
	# Define labels
        # 遍历设置图像对应的label路径
        # (./my_yolo_dataset/train/images/2009_004012.jpg) -> (./my_yolo_dataset/train/labels/2009_004012.txt)
        self.label_files = [x.replace("images", "labels").replace(os.path.splitext(x)[-1], ".txt")
                            for x in self.img_files]
		
	#单GPU的时候rank = -1,主进程是0
	# tqdm库会显示处理的进度
	# 读取每张图片的size信息
	if rank in [-1, 0]:
		image_files = tqdm(self.img_files, desc="Reading image shapes")
		
	# 如果为ture，训练网络时，会使用类似原图像比例的矩形(让最长边为img_size)，而不是img_size x img_size
        # 注意: 开启rect后，mosaic就默认关闭
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
	    
	# 按照高宽比例进行排序，这样后面划分的每个batch中的图像就拥有类似的高宽比
        irect = ar.argsort()
        # 根据排序后的顺序重新设置图像顺序、标签顺序以及shape顺序
		
		
	# 获取第i个batch中，最小和最大高宽比
	mini, maxi = ari.min(), ari.max()

	# 如果高/宽小于1(w > h)，将w设为img_size
	if maxi < 1:
		shapes[i] = [maxi, 1]
	# 如果高/宽大于1(w < h)，将h设置为img_size
	elif mini > 1:
		shapes[i] = [1, 1 / mini]

	确定比例-->乘上512-->取32整数倍

	# 计算每个batch输入网络的shape值(向上设置为32的整数倍)
        self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32. + pad).astype(np.int) * 32
		
	每张图片保持长宽比，将最大边长（w）缩放到指定的img_size大小
		
		
	nm, nf, ne, nd = 0, 0, 0, 0  # number mission, found, empty, duplicate
        # 这里分别命名是为了防止出现rect为False/True时混用导致计算的mAP错误
        # 当rect为True时会对self.images和self.labels进行从新排序
        if rect is True:
            np_labels_path = str(Path(self.label_files[0]).parent) + ".rect.npy"  # saved labels in *.npy file
        else:
            np_labels_path = str(Path(self.label_files[0]).parent) + ".norect.npy"
				
	建数组
	self.imgs = [None] * n  # n为图像总数

	记录缓存图片占用RAM大小
	gb += self.imgs[i].nbytes  # 用于记录缓存图像占用RAM大小
	if rank in [-1, 0]:
		pbar.desc = "Caching images (%.1fGB)" % (gb / 1E9)

	如果使用多GPU训练并开启cache_images时，每个进程都会缓存一份



### 6  yolov4：

https://www.bilibili.com/video/BV1Q54y1D7vj?p=4

https://github.com/bubbliiiing/yolov4-pytorch

[睿智的目标检测30——Pytorch搭建YoloV4目标检测平台](https://blog.csdn.net/weixin_44791964/article/details/106214657)

[睿智的目标检测26——Pytorch搭建yolo3目标检测平台](https://blog.csdn.net/weixin_44791964/article/details/105310627)

**1 主干特征提取网络：**

	DarkNet53 => CSPDarkNet53
		
		CSPnet结构并不算复杂，就是将原来的残差块的堆叠进行了一个拆分，拆成左右两部分：
			
		主干部分继续进行原来的残差块的堆叠；
			
		另一部分则像一个残差边一样，经过少量处理直接连接到最后。
	
	使用Mish激活函数:将DarknetConv2D的激活函数由LeakyReLU修改成了Mish
	
**2  特征金字塔：SPP、PAN**
		
	SPP；3个不同大小的pool（5 9 13） + 1个直连短接边 --->concat堆叠
		
	PAN: 上采样 + 下采样 + 特征融合
			
			P3---conv --- concat + conv X 5 -------------------> yolo head（conv 3x3 + conv 1X1,52*52*75)
							  | 上行                  | 下行
						 conv + upsampling         downsampling(conv实现下采样)
							  |                       |
			P4---conv --- concat + conv X 5 ------>concat + conv X 5 -----> yolo head （conv 3x3 + conv 1X1,26*26*75)
	|						  |                       |
	spp					 conv + upsampling          downsampling
	|						  |                       |
	P5-----——--------------------------------------- concat + convX5 ---> yolo head (conv 3x3 + conv 1X1,13*13*75)               

**3  yolo head**

    3X3 conv （卷积 标准化 激活）

    1X1 conv（只有卷积）
	
	3*（5 + num_classes）:3是3个先验框

	训练过程： featuremap计算loss

		GT映射到featuremap图(13 26 52)------->
						      |
		anchor映射到featuremap图 --------->根据偏移计算box --->根据GT和box大小，计算IOU -->区分正负样本、计算loss
	
    DecodeBox 解码过程：anchor映射到feature map图(中心映射) ---->根据偏移x y w h计算预测出来的box --> 映射回原图---> non_max_suppression（筛选出一定区域内得分最大的方框）
    
    sigmod:让每个物体由左上角网格点预测	
	
	
	生成先验框：先生成中心--->再生成宽高
	
	#----------------------------------------------------------#
	#   生成网格，先验框中心，网格左上角 
	#   batch_size,3,13,13
	#----------------------------------------------------------#
	grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
		batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
	grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
		batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
    
    
    
**4  训练用到的小技巧**

    Mosaic数据增强、Label Smoothing平滑、CIOU、学习率余弦退火衰减
	
    mosaic利用了四张图片,在BN计算的时候一下子会计算四张图片的数据。
		
			读入图片-->分别增强，四个角摆放-->图片的组合和框的组合(merge_bboxes)
	
			get_random_data_with_Mosaic-->merge_bboxes
	
    CIOU: I  G  D  C
	
		ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal,min = 1e-6)
		
		v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0]/torch.clamp(b1_wh[..., 1],min = 1e-6)) - torch.atan(b2_wh[..., 0]/torch.clamp(b2_wh[..., 1],min = 1e-6))), 2)
		alpha = v / torch.clamp((1.0 - iou + v),min=1e-6)
		
    标签平滑：

		def smooth_labels(y_true, label_smoothing,num_classes):
			return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes
	
	
**5   源码**

    预测过程：

    detect_image: letterbox(加padding不失真resize)-->转numpy，归一化,转tensor-->送入网络，得到预测结果-->NMS预测 -->yolo_correct_boxes：回映射去掉灰条-->绘制目标框和得分

    训练自己模型：

    VOCdevkit： voc_annotation.py 
	
    #------------------------------------------------------#
    #   Yolov4的tricks应用
    #   mosaic 马赛克数据增强 True or False 
    #   实际测试时mosaic数据增强并不稳定，所以默认为False
    #   Cosine_scheduler 余弦退火学习率 True or False
    #   label_smoothing 标签平滑 0.01以下一般 如0.01、0.005
    #------------------------------------------------------#
    mosaic = False
    Cosine_lr = False
    smoooth_label = 0
	
    自己的模型：修改yolo.py中的model_path和classes_path，
	
	train.py: anchor_path、YOLOLoss
	
		冻结部分--->解冻训练：
		for param in model.backbone.parameters():
			param.requires_grad = True
	
	先验框长宽比计算:kmeans_for_anchor.py
		
		kmeans:聚类9个中心， 1-IOU(重合程度) = 偏移程度
		
**6  loss 计算**

	三个层计算，每个都类似，loss计算是针对特征层featuremap来讲的
	
	乘3是因为有3个先验框
	
		#-------------------------------------------------#
		#   此时获得的scaled_anchors大小是相对于特征层的
		#-------------------------------------------------#
		scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

		prediction = input.view(bs, int(self.num_anchors/3),
							self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

		# 先验框的中心位置的调整参数
		x = torch.sigmoid(prediction[..., 0])
		y = torch.sigmoid(prediction[..., 1])
		# 先验框的宽高调整参数
		w = prediction[..., 2]
		h = prediction[..., 3]
		# 获得置信度，是否有物体
		conf = torch.sigmoid(prediction[..., 4])
		# 种类置信度
		pred_cls = torch.sigmoid(prediction[..., 5:])
    
	get_target：计算网络应该输出的正确值，对真实框进行判断，找到对应的网格点和先验框，
	
	    计算出正样本在特征层上的中心、宽高 ->正样本属于特征层的哪个grid-> 计算重合度最大的先验框 -->计算tx ty tw th-->大目标loss权重小，小目标loss权重大
	
 	get_ignore:获得哪些样本是应该忽略的。
		
	预测框和真实框对比，获得CIOU loss，或者先验框内部是否包含物体的loss,物体种类的loss
  
  

###  7  yolov5 解读


[YOLOv5代码详解（train.py部分）](https://blog.csdn.net/mary_0830/article/details/107076617)

[YOLOv5代码详解（yolov5l.yaml部分）](https://blog.csdn.net/mary_0830/article/details/107124459)


#### yaml解读

s->m->l->x      smlx

nc：类别数，你的类别有多少就填写多少。从1开始算起，不是0-14这样算。

depth_multiple：控制模型的深度。

width_multiple：控制卷积核的个数。

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]           控制宽度
            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # number of repeats，控制深度
                n = 1

#### 训练代码

**1 输入端：**

Mosaic数据增强、自适应锚框计算、自适应图片缩放

	noautoanchor：但Yolov5中将此功能嵌入到代码中，每次训练时，自适应的计算不同训练集中的最佳锚框值。

	letterbox函数中进行了修改，对原始图像自适应的添加最少的黑边。

**2 Backbone：**

Focus结构：切片操作，类似passthrough

CSP结构：CSP1_X结构应用于Backbone主干网络，另一种CSP2_X结构则应用于Neck中

**3 Neck：**

FPN+PAN结构

**4 Prediction：**

GIOU_Loss，DIOU_nms

大图小目标检测：

对大分辨率图片先进行分割，变成一张张小图，再进行检测。为了避免两张小图之间，一些目标正好被分割截断，所以两个小图之间设置overlap重叠区域，比如分割的小图是960*960像素大小，则overlap可以设置为960*20%=192像素。


ModelEMA：

	Model Exponential Moving Average，近期数据更高权重的平均方法

	[指数移动平均（EMA）的原理及PyTorch实现](https://www.jianshu.com/p/f99f982ad370)

	模型权重在最后的n步内，会在实际的最优点处抖动，所以我们取最后n步的平均，能使得模型更加的鲁棒

DDP训练：

	# DP mode
	if cuda and rank == -1 and torch.cuda.device_count() > 1:
		model = torch.nn.DataParallel(model)

	# SyncBatchNorm
	if opt.sync_bn and cuda and rank != -1:
		model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
		logger.info('Using SyncBatchNorm()')

	主进程，rank 等于0或者-1

混合精度训练: scaler = amp.GradScaler(enabled=cuda)



