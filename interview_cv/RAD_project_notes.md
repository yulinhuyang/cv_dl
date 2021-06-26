RAD  功耗异常检测

###  RAD  功耗异常检测

[Python特征选择(全)](https://zhuanlan.zhihu.com/p/348201771)

[Distance correlation（距离相关系数）](https://blog.csdn.net/jiaoaodechunlv/article/details/80655592)

[特征工程之距离相关系数（ Distance correlation coefficient ）]()


**资源包括：**

	WIFI蓝牙、CPU、GPU、GNSS、Audio 等100多个器件的指标

	每个器件的资源占用：usage ms(用量)、功耗 mAs

	moderm和WIFI：流量 kb/时长
	
	没有label，进行统计来标注label
	
**数据标注：**

	X：100多个特征，Y:CPU的后台功耗数据
	
	数据处理：脱敏，一个用户一个APP 每隔24小时上报一次，一共180个左右的APP。
	
	数据归一化(减去均值/方差) ---->spearman/person相关性分析 ----> 回归树分析 ---->百分位数标记label
	
	特征筛选：
	
		spearman/person: 0.3~0.9之间是正常的。取相关性前20的，筛选出50个特征。(0.3< Ps <0.85) (0.3< Pd <0.85)
		
		不同的APP筛选出来的特征的维度，略有差异，取并集。
		
		Pearson相关用于双变量正态分布的资料。用于计算数值特征两两间的相关性，数值范围[-1，1]。

		Spearman秩相关，当两变量不符合双变量正态分布的假设时，需用Spearman秩相关来描述变量间的相互变化关系。

		线性关系可以通过pearson相关系数来描述，单调关系可以通过spearman或者kendall来描述，非线性如何描述，距离相关系数可以非线性相关性。

		方差膨胀因子也称为方差膨胀系数（Variance Inflation），用于计算数值特征间的共线性，一般当VIF大于10表示有较高共线性。
			
	回归树cart分析：
	
		最多四层深；一个节点对应一个特征进行分离。分离的停止条件，每个叶子的样本数，小于样本总数的10%
		
		停止后，认为每个叶子节点同分布，基本属于具有相同的用户习惯。
		
	百分位数标记label:
		
		同一个叶子节点的，取百分位数，分四种情况：极端（0.5%）、异常（2%）、超用量（10%）、正常
		
		超级APP:微信，0.2%作为极端异常。

**异常定义方式：**

	宁错过，不误杀

	1  百分位数的门限

	2  3 sigma法则， > 均值 u + 3* sigma

	3  上四分位数+ 3* 四分位数间隔： 25%  50% 75% ， 阈值> 75% + (75%-25%) 的值
	
**模型优化：**

	超参搜索：矮胖的5层，变成了高瘦的6层

	loss: BCE loss，先用sigmod处理输入，再送入

	[Pytorch详解BCELoss和BCEWithLogitsLoss](https://blog.csdn.net/qq_22210253/article/details/85222093)
	
``` python

m = nn.Sigmoid()

loss = nn.BCELoss(size_average=False, reduce=False)
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
lossinput = m(input)
output = loss(lossinput, target)

```	

      label onehot编码：不同产品+ 不同APP + CPUBG 一种情况
      
      数据复制：不平衡问题的处理，或者loss加入权重


