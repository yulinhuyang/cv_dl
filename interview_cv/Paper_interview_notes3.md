主要参考：

Bubbliiiing、霹雳吧啦Wz、江大白


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
		

yolov4：

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
  
  
  
