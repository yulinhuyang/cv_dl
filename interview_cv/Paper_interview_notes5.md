
### 1 sort与deepsort

[Deep SORT多目标跟踪算法代码解析(上)](https://zhuanlan.zhihu.com/p/133678626)

[关于 Deep Sort 的一些理解](https://zhuanlan.zhihu.com/p/80764724)

[基于深度学习的目标跟踪(Yolov3+deepsort)](https://blog.csdn.net/qq_38109843/article/details/89457442)

https://github.com/nwojke/deep_sort

https://github.com/pprp/deep_sort_yolov3_pytorch

https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch


#### 1.1  关于yolo + sort 

Detections是通过目标检测器得到的目标框，Tracks是一段轨迹。核心是匹配的过程与卡尔曼滤波的预测和更新过程。

	目标检测器得到目标框Detections---->
					  |
	卡尔曼滤波器预测当前的帧的Tracks---->将Detections和Tracks进行IOU匹配----> 得到Unmatched Tracks(deltede)、Matched Track、Unmatched Detections
		 |     |                                                                                                |                 |
		 |   卡尔曼滤波可以根据Tracks状态预测下一帧的目标框状态。<------------------------------------------------                  |  
		 |________________________________________________________________________________________________________________| new track     


匈牙利匹配：二分图遍历，回溯





#### 1.2 关于yolo + deepsort

Deep SORT算法在SORT算法的基础上增加了级联匹配(Matching Cascade)+ 新轨迹的确认(confirmed)。总体流程就是：

	卡尔曼滤波器预测轨迹Tracks

	使用匈牙利算法将预测得到的轨迹Tracks和当前帧中的detections进行匹配(级联匹配和IOU匹配)

	卡尔曼滤波更新。

Matching Cascade：计算相似度矩阵的方法使用到了外观模型(ReID)和运动模型(马氏距离)来计算相似度

                                                                 ___
	运动模型(马氏距离) lambda d1 ------                      |   gating_threshold
					   |---> cost matrix <---
					   |      |              |__ maxing distance
	cosine Distance: (1-lambda)Xd2 ----       |
						  |
						  |
						  |
						  |
		 ---- Detections---		  |
		|		  |--------->	匈牙利匹配			  
		|				  |				 
	    |--confirmed tracked                  |
	    |   (missing age =0 )                 |
	    |                                     |
	    |___________miss age+=1 ---------------
			    until max_age

 
 级联匹配的数据关联步骤，匹配过程是一个循环(max age个迭代，默认为70)，也就是从missing age=0到missing age=70的轨迹和Detections进行匹配，没有丢失过的轨迹优先匹配，
 
 丢失较为久远的就靠后匹配。通过这部分处理，可以重新将被遮挡目标找回，降低被遮挡然后再出现的目标发生的ID Switch次数。
 
 
ReID模块: 提取表观特征,生成了128维的embedding

Track模块，轨迹类，用于保存一个Track的状态信息

Tracker模块，Tracker模块掌握最核心的算法，卡尔曼滤波和匈牙利算法都是通过调用这个模块来完成的。

NearestNeighborDistanceMetric:

马氏距离计算物体检测Bbox dj和物体跟踪BBox yi之间的距离



#### 1.3 总结：

**1.使用级联匹配算法**

针对每一个检测器都会分配一个跟踪器，每个跟踪器会设定一个time_since_update参数。如果跟踪器完成匹配并进行更新，那么参数会重置为0，否则就会+1。

实际上，级联匹配换句话说就是不同优先级的匹配。在级联匹配中，会根据这个参数来对跟踪器分先后顺序，参数小的先来匹配，参数大的后匹配。

也就是给上一帧最先匹配的跟踪器高的优先权，给好几帧都没匹配上的跟踪器降低优先权（慢慢放弃）。至于使用级联匹配的目的，我引用一下博客②里的解释：

**2.添加马氏距离与余弦距离：全面的差异性衡量**

马氏距离实际上是针对运动信息与外观信息的计算.针对于位置进行区分。

余弦距离则是一种相似度度量方式,针对于方向。

**3.添加深度学习特征**

改进中加入了一个深度学习的特征提取网络，所有confirmed的追踪器（其中一个状态）每次完成匹配对应的detection的feature map存储进一个list（存储的数量100帧）。在每次匹配之后都会更新这个feature map的list，比如去除掉一些已经出镜头的目标的特征集，

保留最新的特征将老的特征pop掉等等。这个特征集在进行余弦距离计算的时候将会发挥作用

4.IOU与匈牙利算法匹配：尽量多


### 2 PFLD

主干： MobilenetV2 修改

数据不平衡：训练过程加入人脸几何约束geometric constraint，使得大角度，难样本，传递更大的loss。

加入人脸属性信息（profile-face, frontal-face, head-up, head-down, expression,and occlusion），解决数据不平衡data imbalance。

辅助网络用以监督 PFLD 网络模型的训练,对每一个输入的人脸样本进行三维欧拉角估计,作为区分数据分布的依据。

### 3 transformer


RNN 记忆长度有限，因此有了LSTM

无法并行化

可以记忆长度无限长，可以并行化


**self-attention**

		|---->Wv  ---> v
	a ----->  Wq  ---> q
		|---->Wk  ----> k
		

	q: query(to match others)

	k: key (to be matched)

	v: information to be extracted


	q与k进行match，scaled Dot-Product Attention:

	q点乘k

	进行点乘后的数值很大，导致通过softmax后的梯度变的很小。

	Attention（Q,K,V）= softmax(Q * Ktrans / d)*V

	Q和K的点乘结果，代表对V的关注程度


**multi-head self-attention**

	将q进行均分，K进行均分，v进行均分
	
	multihead(Q,K,V) = concat(head1,...,headn)*Wo
	
	where headi= Attention(Q*Wqi,K*Wki,V*Wvi)
	
	
	按照head个数进行均分，分为head1 ,head2-->	对于拆分后的每个head,分别为head1、head2，分别进行self-attention的操作 -->拼接每个head得到的结果 ——> Wo对拼接之后的数据进行进一步融合
	
	和组卷积比较像
	
	
	poition Encoding 位置编码
