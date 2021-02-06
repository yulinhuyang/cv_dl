# 1. 监督学习

## 1.1. 广义线性模型

岭回归

Lasso

弹性网络

最小角回归

正交匹配追踪法（OMP）

贝叶斯回归

logistic 回归与最大熵模型

Passive Aggressive Algorithms（被动攻击算法）

多项式回归：用基函数展开线性模型

线性模型选择与正则化

非线性模型

## 1.2. 线性和二次判别分析

使用线性判别分析来降维

LDA

QDA

Shrinkage（收缩）

## 1.3. 内核岭回归

## 1.4. 支持向量机

SVC

SVM

SVR

密度估计, 异常（novelty）检测

核函数

实现细节


## 1.5. 随机梯度下降

## 1.6. 最近邻

无监督最近邻

最近分类回归

最近质心分类

邻域成分分析

## 1.7. 高斯过程

高斯过程回归（GPR）   高斯过程分类（GPC） 高斯过程内核

## 1.8. 交叉分解

## 1.9. 朴素贝叶斯

多项分布朴素贝叶斯   伯努利朴素贝叶斯

## 1.10. 决策树

决策树算法: ID3, C4.5, C5.0 和 CART

## 1.11. 集成方法

集成学习   提升方法

Bagging meta-estimator（Bagging 元估计器）

AdaBoost

Gradient Tree Boosting（梯度树提升）

Voting Classifier（投票分类器）

Voting Regressor(投票回归器)

XGBoost的模型

树的生成:		

1  判断切分增益，Gain值越大，说明分裂后能使目标函数减少越多，就越好分裂查找算法		

2  Basic Exact Greedy Algorithm （精确贪心算法）

Approximate Algorithm（近似算法）

加权分位数略图（Weighted Quantile Sketch）	

稀疏值处理（Sparsity-aware Split Finding）

分块并行（Column Block for Parallel Learning）	

缓存访问（Cache-aware Access）		

核外"块计算（Blocks for Out-of-core Computation）

## 1.12. 多类和多标签算法

多标签分类格式、1对其余 

多输出回归分类

链式分类器

## 1.13. 特征选择

移除低方差特征

单变量特征选择

使用 SelectFromModel 选取特征

特征选取作为 pipeline（管道）的一部分

## 1.14. 半监督学习



## 1.15. 等式回归

## 1.16. 概率校准

## 1.17. 神经网络模型（有监督）

多层感知器 

正则化、算法

使用 warm_start 的更多控制

# 2. 无监督学习

## 2.1. 高斯混合模型

高斯混合GMM

变分贝叶斯高斯混合

## 2.2. 流形学习

Isomap

局部线性嵌入

改进型局部线性嵌入（MLLE）

黑塞特征映射（HE）

谱嵌入

局部切空间对齐（LTSA）

多维尺度分析（MDS）

t 分布随机邻域嵌入（t-SNE）

## 2.3. 聚类

聚类方法

K-means

Affinity Propagation

Mean Shift

Spectral clustering

层次聚类

DBSCAN

OPTICS

## 2.4. 双聚类

Spectral Co-Clustering

Spectral Biclustering

## 2.5. 分解成分中的信号（矩阵分解问题）

主成分分析（PCA）

奇异值分解

潜在语义分析

概率潜在语义分析

截断奇异值分解和隐语义分析

词典学习

因子分析

独立成分分析（ICA）

非负矩阵分解(NMF 或 NNMF)

隐 Dirichlet 分配（LDA）

## 2.6. 协方差估计

经验协方差

收缩协方差

稀疏逆协方差

鲁棒协方差估计

## 2.7. 新奇和异常值检测

离群点检测方法一览

Novelty Detection（新奇点检测）

Outlier Detection（离群点检测）

使用LOF进行新奇点检测

离群点检测方法一览

## 2.8. 密度估计

密度估计: 直方图

核密度估计

## 2.9. 神经网络模型（无监督）

# 3. 模型选择和评估

## 3.1. 交叉验证：评估估算器的表现

## 3.2. 调整估计器的超参数

网格追踪法–穷尽的网格搜索：GridSearchCV 

随机参数优化：RandomizedSearchCV 

参数搜索技巧：指定目标度量、为评估指定多个指标、并行机制、对故障的鲁棒性

暴力参数搜索的替代方案：模型特定交叉验证、出袋估计

## 3.3. 模型评估: 量化预测的质量

scoring 参数: 定义模型评估规则

分类指标：

混淆矩阵

汉明损失：hamming_loss

精准，召回和 F-measures

多类和多标签分类

Jaccard 相似系数 score

Hinge loss

Log 损失

马修斯相关系数

多标记混淆矩阵

Receiver operating characteristic (ROC)

零一损失：zero_one_loss

多标签排名指标：排序损失

回归指标：mean_squared_error, mean_absolute_error, explained_variance_score 和 r2_score

聚类指标：

## 3.4. 模型持久化

 pickle

## 3.5. 验证曲线: 绘制分数以评估模型

# 4. 检验

## 4.1. 部分依赖图

# 5. 数据集转换

## 5.1. Pipeline（管道）和 FeatureUnion（特征联合）: 合并的评估器

Pipeline: 链式评估器

回归中的目标转换

FeatureUnion（特征联合）: 复合特征空间

用于异构数据的列转换器

## 5.2. 特征提取

从字典类型加载特征

特征哈希（相当于一种降维技巧）

文本特征提取

图像特征提取

## 5.3 预处理数据

标准化

非线性转换

归一化

类别特征编码

离散化

缺失值补全

生成多项式特征

自定义转换器

## 5.4 缺失值插补

单变量与多变量插补

单变量插补

标记缺失值

## 5.5. 无监督降维

PCA: 主成份分析

随机投影

特征聚集

## 5.6. 随机投影

Johnson-Lindenstrauss 辅助定理

高斯随机投影

稀疏随机矩阵

## 5.7. 内核近似

内核近似的 Nystroem 方法

径向基函数内核

加性卡方核

Skewed Chi Squared Kernel 

## 5.8. 成对的矩阵, 类别和核函数

余弦相似度

线性核函数

多项式核函数

Sigmoid 核函数

RBF 核函数

拉普拉斯核函数

卡方核函数

## 5.9. 预测目标 (y) 的转换

# 6. 数据集加载工具

## 6.1. 通用数据集 API

## 6.2. 玩具数据集

## 6.3 真实世界中的数据集

## 6.4. 样本生成器

## 6.5. 加载其他数据集

# 7. 使用scikit-learn计算

## 7.1. 大规模计算的策略: 更大量的数据

## 7.2. 计算性能

## 7.3. 并行性、资源管理和配置

