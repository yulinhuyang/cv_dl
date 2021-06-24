

### 不同机型的阈值表

[Python特征选择(全)](https://zhuanlan.zhihu.com/p/348201771)

[Distance correlation（距离相关系数）](https://blog.csdn.net/jiaoaodechunlv/article/details/80655592)

[特征工程之距离相关系数（ Distance correlation coefficient ）]()


**特征筛选：**

	K features,one Y  ---> person、speraman、distance相关系数--—--—---—---> 方差膨胀因子VIF ------------> LASSO：多元线性回归 --------------
															   k1 feature                  k2 feature  |                                    |
	数据量纲归一化			0.3< Ps <0.85                                                              |———— Pearson Partial Correlation ----------筛选的特征
							0.3< Pd <0.85
							Pp >= 0.3 & |Pp-Ps|< 0.1

**数据标注：**

	Bucketize划分数据 ---> 计算条件分位数 ---> Estimate joint tail probability


96种特征----> 50种特征

Pearson相关用于双变量正态分布的资料。用于计算数值特征两两间的相关性，数值范围[-1，1]。

Spearman秩相关，当两变量不符合双变量正态分布的假设时，需用Spearman秩相关来描述变量间的相互变化关系。

线性关系可以通过pearson相关系数来描述，单调关系可以通过spearman或者kendall来描述，非线性如何描述，距离相关系数可以非线性相关性。

方差膨胀因子也称为方差膨胀系数（Variance Inflation），用于计算数值特征间的共线性，一般当VIF大于10表示有较高共线性。

模型相关：


    CPUBG    CPUFG 

    ALARM    WAKELOCK(唤醒锁定)

模型结构：6层 全连接

多标签分类：sigmod 函数，和不是1 

	sigmoid应该会将logits中每个数字都变成[0,1]之间的概率值，假设结果为[0.01, 0.05, 0.4, 0.6, 0.3, 0.1, 0.5, 0.4, 0.06, 0.8], 然后设置一个概率阈值，比如0.3，如果概率值大于0.3，则判定类别符合，那么该输入样本则会被判定为类别3、类别4、类别5、类别7及类别8。即一个样本具有多个标签。
	在这里强调一点：将sigmoid激活函数应用于多标签分类时，其损失函数应设置为binary_crossentropy。



