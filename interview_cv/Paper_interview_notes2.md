
### PFID: 

https://zhuanlan.zhihu.com/p/73546427

1  损失函数： 对于样本量比较大的数据（如正脸，即欧拉角都相对较小的情况），给予一个小的权值，在进行梯度的反向传播的时候，对模型训练的贡献小一些；

对于样本量比较少的数据（侧脸、低头、抬头、表情极端），给予一个较大的权值。

2  配合训练的子网络

对样本进行三维欧拉角的估计，用于区分数据分布，主要是为了服务关键点检测网络



### yolov3 整合

1  backbone修改，换成Daknet19 -->Darknet53
	
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
  
4 正负样本匹配

   使用logistic regression预测

5  损失计算：lambda1*置信度损失Lconf+ lambda2*分类损失Lcls+ lambda3*定位损失
   
   置信度损失：sigmoid（C）--> BCE(二值交叉熵损失)
   
	   Oi:预测目标边界框与真实目标边界框的IOU 
	   
	   ci通过sigmoid函数得到的预测置信度,N正负样本格式
	
   类别损失：二值交叉熵损失，经过sigmoid处理sigmoid(Cij)  
   
   定位损失：误差平方和计算L2损失

   
### yoloV3 SPP  yoloV3 SPP-ultralytics

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
   
 
### 目标检测

tensorflow官方API 

https://github.com/tensorflow/models/tree/master/research

https://github.com/tensorflow/models/tree/master/research/object_detection

mask-rcnn 检测+抠出来


**PASCAL VOC数据集**

	develop kit    20类

	voc2012的test数据集不公开

	voc2007的test已经公开了,可以用作测试集

	每一类都是都是有训练和验证 bus_train bus_val bus_trainval

	标注软件；https://github.com/tzutalin/labelImg

**MS COCO数据集**

	https://blog.csdn.net/qq_37541097/article/details/112248194

	object 80类

	stuff类别：包含了没有明确边界的材料和对象

	object 80是stuff91类的子类

	包含的VOC的全部类别
	
	COCO上预训练效果更好，但更费时
	
    cocodataset.org
   
2017 train images 训练文件 2017 val images 验证文件 2017 train/val images 标注文件

	一般情况下train和val就够了，因为val和test数据分布是一样的。大型比赛中，划分test是有意义的。
	
	cocodataset.org/#format-data
	
	json文件load : images、annotations 
	
	使用pycocotools.coco读取
	
	coco_classes = dict([(v['id']，v['name']) for k,v in coco.cats.items()])
	
	image_id --> ann_id --> target标注信息 --> file_name
	
	https://cocodataset.org/#home
	

**目标检测指标**
	
	TP: IOU>0.5的检测框数量（同一个GT只计算一次）
	
	FP: IOU<0.5的检测框()
	
	FN：没有检测的GT数量，漏检
	
	precision:查准率。TP/（TP+FP）模型预测的所有目标中，预测正确的比例
	
	recall:查全率。TP/ (TP+FN)

	AP：P-R曲线下面积
	
	P-R曲线：precision -recall曲线
	
	mAP：多个类别取均值
	
	按照conf大小，排序汇总表格 ——>按照conf大小，逐个填写计算recall precision --> 合并recall相同的，precision取大--> 计算MAP
	
	precision 为横坐标，recall为纵坐标，得到PR曲线，P-->Y, R-->x

    https://cocodataset.org/#detection-eval 
	
	mAP@0.5：mean Average Precision（IoU=0.5）
    
	即将IoU设为0.5时，计算每一类的所有图片的AP，然后所有类别求平均，即mAP
	
	mAP@.5:.95（mAP@[.5:.95]）
    
	表示在不同IoU阈值（从0.5到0.95，步长0.05）（0.5、0.55、0.6、0.65、0.7、0.75、0.8、0.85、0.9、0.95）上的平均mAP。
	
	coco evaluation Result：
		
		关注： AP 0.5:0.95，AP 0.5
		
		APsmall/APmedium 大小目标
		
		Average Recall:一张图片检测的最多目标数	
	
**目标检测基础**
   
     one-stage： SSD yolo
		
		基于anchors直接进行分类以及调整边界框
	 
	 Two-stage: faster rcnn，
		
		1  通过专门模块生成候选框(RPN)，寻找前景以及调整边界框（基于anchors）
		
		2  基于之前生成的候选框进行进一步分类以及调整边界框（基于proposals）
	
### Faster RCNN 

#### 1  R-CNN

	Region with CNN feature
	
	selective search(一图生成2K个候选框region proposal) -->每个候选框进行深度网络特征提取-->特征送入每一类的SVM分类器，判断是否属于该类-->回归器精修候选框位置
	
	NMS:寻找得分最高的目标-->计算其他目标与该目标的IOU值 --> 删除所有IOU值大于阈值的目标
	
#### 2  Fast RCNN

    selective search(一图生成2K个候选框region proposal) —> 整张图上，用网络提取特征，特征不重复计算--> ROI pooling缩放为7X7大小的特征图，然后将特征图展平通过一系列全连接层得到结果。
	
	正负样本采样：对SS候选框采样，64 from 2000,IOU > 0.5 正样本
	
    ROI pooling: 7X7 划分特征图，不限制输入图像的尺寸。
	
	分类器：ROI feature vector ,输出N+1（1为背景）个类别的概率
	
	回归器：（N+1）*4

    边界框回归器、损失函数
	
	 
	交叉熵损失在二分类情况下，使用sigmoid输入，各个输出节点之间互不相干。
	          |         
			  在多分类情况下，使用softmax输出，所有输出概率和为1
	
	loss；分类损失（交叉熵） + [艾佛森括号]*边界框回归损失
	
#### 3  Faster RCNN = RPN + Fast R-CNN

RPN(region proposal network):生成候选框：
	
	在feature map上进行3X3滑窗 ——> 产生256d特征（VGG16） --> cls layer 生成2K scores 区分前景和背景 和reg layer 生成4K coordinates 检测框
	
	对于特征图上的每3X3的滑动窗口，计算出滑动窗口中心对应的原始图像上的中心点,回映射原图 --> 在原图上计算出k个anchor boxes（注意与proposal差异）

    anchor: 三种尺度(面积){128,256,512} 三种比例{1:1,1:2,2:1}，每个位置在原图上都对应有3X3=9个anchor
	
	通过一个小的感受野预测一个大的目标边界框是可以的。
	
	感受野计算: F(i) = (F(i+1)-1)*stride + Ksize  
	
	区分：利用RPN生成的边界框回归参数，将anchor调整为候选框
	
    正负样本采样：每张图片采样256个anchor,采样正负样本比例1:1.正样本不足128，负样本填充。
	
	正样本： IOU > 0.7,或者取与GT有最大IOU
	
	负样本：anchor与所有的GT的IOU < 0.3
	
	RPN multi-task loss：  cls分类损失（BCE 二值交叉熵）+ 边界框reg回归损失
	
	    Nreg:anchor位置的个数
	
	    使用二值交叉熵损失的时候，预测k个scores，区分二值交叉熵损失和多分类交叉熵损失
	
训练：RPN loss + faster R-CNN  loss相加


图像输入网络得到特征图--->RPN生成候选框--->候选框投影到特征图上得到特征矩阵-->将每个特征矩阵通过ROI pooling层缩放到7X7大小的特征图-->接着讲特征图展平通过一系列全连接层得到预测结果。

对比和区分RPN的anchor和head的anchor


理解： 由于proposal是对应MxN尺度的，所以首先使用spatial_scale参数将其映射回(M/16)x(N/16)大小的feature map尺度；再将每个proposal对应的feature map区域水平分为 [公式] 的网格；

对网格的每一份都进行max pooling处理。

https://zhuanlan.zhihu.com/p/31426458
      
#### 4  FPN结构 feature pyramid networks

   先top --> down
   
   再down --> top
   
   对不同特征图上的特征进行融合，然后再进行预测
				 |
				2x up
				 |
  -->1x1conv---> +
                 |
				 
	1x1 conv 调整channel

	P6只用于RPN部分，不在fast-rcnn部分使用
	
	针对不同的预测特征层，RPN 和fast RCNN的权重共享
	
	不同的预测特征层对应不同的面积，P2->P6,对应 32*32、64*64、128*128、256*256、512
		
	RPN得到的一系列proposal,如何映射到不同的特征层上？ ---> 计算分配
	
#### 5  SSD网络
   
   faster rcnn问题：小目标不好，慢

   SSD：在不同特征尺度上预测不同尺度的目标

   default box的scale和aspect设定：3个层是每个featmap点生成4个default box 和3个层是6个 default box，一共是8732个
   
	   default box的设定示例
	   
   predictor预测: （C+4）X k = C x k  + 4 X k

	   C Xk：背景、类1 、类2...
	   
	   在featuremap上的每个位子都会生成k个default box
	   
	   对比fastrcnn中，是4 * c * k 个，ssd是4k个
	   
   正负样本的选择：
		
		正样本：与GT的IOU值最大，或与GT IOU > 0.5 
		
		hard negative mining: 使用最大confidence loss 排序，选择负样本
		
		负正样本比：3:1
		
	损失：Lconf类别损失(正样本损失+负样本损失)+ alpha * Lloc定位损失
	
#### 6 RetinaNet:
  
   focal loss for dense object detection 
   
   one-stage首次超过two-stage

结构：

   没有根据C2生成P2，多了一个P7
   
   3组scale X 3组ratios = 9 个anchor
   
权值共享：

   P3-P7 的预测器class subnet KA 和box subnet 4A（而不是4KA）的权值是共享的
   
回归参数：
   
   FasterRCNN中对于预测特征层上的每一个anchor都会针对每个类别去生成一组边界框回归参数(4A而不是4KA)
 
损失函数：

   正负样本匹配： Iou >=0.5,正样本;IOU < 0.4 负样本;  0.4 <IoU < 0.5丢弃

   Lcls： sigmod Focal loss，所有的正负样本
   
   Lreg : L1 loss，所有正样本的损失
 
#### 7  yolov3 SPP 源码

训练代码：

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
	
	
	isnumeric  判断是否是numeric类型
	

 
