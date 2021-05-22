#### 1  交叉熵损失函数推导


#### 2 SVM 损失函数推导



####  3  CV_interviews 补充

**3 代码实现卷积**

[卷积的三种模式full, same, valid以及padding的same, valid](https://zhuanlan.zhihu.com/p/62760780)

f -->s -->v

full模式的意思是，从filter和image刚相交开始做卷积。

same的意思是，当filter的中心(K)与image的边角重合时，开始做卷积运算。

valid的意思，当filter全部在image里面的时候，进行卷积运算。


**5 python ::-1**

[::-1] 顺序相反操作

[-1] 读取倒数第一个元素

[3::-1] 从下标为3（从0开始）的元素开始翻转读取


**13  为什么max pooling 要更常用？什么场景下 average pooling 比 max pooling 更合适**

讲一下 pooling 的作用， 为什么 max pooling 要更常用？哪些情况下，average pooling 比 max pooling 更合适？

作用：对输入的特征图进行压缩，一方面使特征图变小，简化网络计算复杂度；一方面进行特征压缩，提取主要特征。

通常来讲，max-pooling 的效果更好，虽然 max-pooling 和 average-pooling 都对数据做了下采样，但是 max-pooling 感觉更像是做了特征选择，选出了分类辨识度更好的特征，提供了非线性。 pooling 的主要作用一方面是去掉冗余信息，一方面要保留 feature map 的特征信息，在分类问题中，我们需要知道的是这张图像有什么 object，而不大关心这个 object 位置在哪，在这种情况下显然 max pooling 比 average pooling 更合适。在网络比较深的地方，特征已经稀疏了，从一块区域里选出最大的，比起这片区域的平均值来，更能把稀疏的特征传递下去。

average-pooling 更强调对整体特征信息进行一层下采样，在减少参数维度的贡献上更大一点，更多的体现在信息的完整传递这个维度上，在一个很大很有代表性的模型中，比如说 DenseNet 中的模块之间的连接大多采用 average-pooling，在减少维度的同时，更有利信息传递到下一个模块进行特征提取。

average-pooling 在全局平均池化操作中应用也比较广，在 ResNet 和 Inception 结构中最后一层都使用了平均池化。有的时候在模型接近分类器的末端使用全局平均池化还可以代替 Flatten 操作，使输入数据变成一位向量。

**16 卷积层相较于全连接层的优势**

卷积层相较于全连接层需要训练的参数更少，所以神经网络的设计离不开卷积层

卷积层通过参数共享和稀疏连接两种方式来保证单层卷积中的训练参数少

**17 网络中常用的损失函数汇总：**

https://zhuanlan.zhihu.com/p/58883095

**18 有哪些修改、调试模型的经验分享**

**19 目标检测评价指标mAP的计算**

见 2020_algorithm_intern_information

**20  实例分割中的评价指标：**

https://blog.csdn.net/weixin_40546602/article/details/105292391 

**21 手动推导反向传播公式BP**

鸡： 计算图

鸡： 激活函数

踢： 梯度下降

连： 链式求导

长：张量

**22 深度分离理解**

[深度可分离卷积](https://zhuanlan.zhihu.com/p/92134485)

深度卷积负责滤波，尺寸为(DK,DK,1)，共M个，作用在输入的每个通道上；逐点卷积负责转换通道，尺寸为(1,1,M)，共N个，作用在深度卷积的输出特征映射上。

**25  batch size 和 learning rate 的关系：** 

https://www.zhihu.com/question/64134994/answer/216895968

通常当我们增加batchsize为原来的N倍时，要保证经过同样的样本后更新的权重相等，按照线性缩放规则，学习率应该增加为原来的N倍[5]。

但是如果要保证权重的方差不变，则学习率应该增加为原来的sqrt(N)倍[7]，目前这两种策略都被研究过，使用前者的明显居多。使用这样的策略，就可以缓解大的batchsize带来的以上问题。

对此实际上是有两个建议：如果增加了学习率，那么batch size最好也跟着增加，这样收敛更稳定。

尽量使用大的学习率，因为很多研究都表明更大的学习率有利于提高泛化能力。如果真的要衰减，可以尝试其他办法，比如增加batch size，学习率对模型的收敛影响真的很大，慎重调整。

**26 类别不均衡问题怎么解决**

https://zhuanlan.zhihu.com/p/56882616

https://blog.csdn.net/vivian_ll/article/details/105201120

**28 权重初始化方法有哪些**

深度学习中神经网络的几种权重初始化方法

Xavier初始化：

Xavier initialization是 Glorot 等人为了解决随机初始化的问题提出来的另一种初始化方法，他们的思想倒也简单，就是尽可能的让输入和输出服从相同的分布，这样就能够避免后面层的激活函数的输出值趋向于0。

虽然Xavier initialization能够很好的 tanH 激活函数，但是对于目前神经网络中最常用的ReLU激活函数，还是无能能力。

He初始化：

提出了一种针对ReLU的初始化方法，一般称作 He initialization。

https://zhuanlan.zhihu.com/p/72374385

**29  计算机视觉中的注意力机制**

https://zhuanlan.zhihu.com/p/146130215

**30  onehot 编码可能会遇到的问题**

onehot适用：使用one-hot编码来处理离散型的数据，使用one-hot编码，将离散特征的取值扩展到了欧式空间，离散特征的某个取值就对应欧式空间的某个点。将离散型特征使用one-hot编码，会让特征之间的距离计算更加合理。

one-hot编码要求每个类别之间相互独立，如果之间存在某种连续型的关系，或许使用distributed respresentation（分布式）更加合适。

One-hot encoding

优点：解决了分类器不好处理分类数据的问题，在一定程度上也起到了扩充特征的作用。它的值只有0和1，不同的类型存储在垂直的空间。

缺点：当类别的数量很多时，特征空间会变得非常大，容易造成维度灾难。

Label encoding

优点：解决了分类编码的问题，可以自由定义量化数字。但其实也是缺点，因为数值本身没有任何含义，只是排序。如大中小编码为123，也可以编码为321，即数值没有意义。

缺点：可解释性比较差。比如有[dog,cat,dog,mouse,cat]，我们把其转换为[1,2,1,3,2]，这里就产生了一个奇怪的现象：dog和mouse的平均值是cat。因此，Label encoding编码其实并没有很宽的应用场景。

定类类型的数据，建议使用one-hot encoding，是纯分类，不排序。定序类型的数据，建议使用label encoding。

使用稀疏向量来节省空间，目前大部分算法支持稀疏向量形式的输入。

连续型数据预处理：二值化 和 分段。

**31  分类问题有哪些评价指标？每种的适用场景**


**32  L1 正则化与 L2 正则化的区别**

https://cloud.tencent.com/developer/article/1456966





