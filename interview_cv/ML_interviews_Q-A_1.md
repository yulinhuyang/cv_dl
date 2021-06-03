
李宏毅笔记： https://github.com/Sakura-gh/ML-notes


**01 LR和SVM的比较**

理解svm 和hinge loss 

https://zhuanlan.zhihu.com/p/49331510

https://www.zhihu.com/question/47746939


KKT条件。(1)是对拉格朗日函数取极值时候带来的一个必要条件，(2)是拉格朗日系数约束（同等式情况），(3)是不等式约束情况，(4)是互补松弛条件

**9  SMO算法**

https://zhuanlan.zhihu.com/p/28660098

https://zhuanlan.zhihu.com/p/29212107

**10 集成学习相关问题**

Bagging和Boosting 的主要区别： 放回、权值、并串行

样本选择上: Bagging采取Bootstraping的是随机有放回的取样，Boosting的每一轮训练的样本是固定的，改变的是买个样的权重。

样本权重上：Bagging采取的是均匀取样，且每个样本的权重相同，Boosting根据错误率调整样本权重，错误率越大的样本权重会变大。

预测函数上：Bagging所以的预测函数权值相同，Boosting中误差越小的预测函数其权值越大。

并行计算: Bagging 的各个预测函数可以并行生成;Boosting的各个预测函数必须按照顺序迭代串行生成。

将决策树与以上框架组合成新的算法

Bagging + 决策树 = 随机森林

AdaBoost + 决策树 = 提升树

gradient + 决策树 = GDBT

boosting:  Improving Weak Classifiers 



