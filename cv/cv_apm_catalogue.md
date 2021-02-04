目录

**第一部分图像生成**

**第1章摄像机的几何模型**

1.1图像成像

1.1.1针孔透视

1.1.2弱透视

1.1.3带镜头的照相机

1.1.4人的眼睛

1.2内参数和外参数

1.2.1刚体变换和齐次坐标

1.2.2内参数

1.2.3外参数

1.2.4透视投影矩阵

1.2.5弱透视投影矩阵

1.3照相机的几何标定

1.3.1使用线性方法对照相机进行标定

1.3.2使用非线性方法对照相机进行标定

1.4注释

习题

编程练习

**第2章光照及阴影**

2.1像素的亮度

2.1.1表面反射

2.1.2光源及其产生的效果

2.1.3朗伯+镜面反射模型

2.1.4面光源

2.2阴影的估算

2.2.1辐射校准和高动态范围图像

2.2.2镜面反射模型

2.2.3对亮度和照度的推理

2.2.4光度立体技术:从多幅阴影图像恢复形状

2.3对互反射进行建模

2.3.1源于区域光在一个块上的照度

2.3.2热辐射和存在性

2.3.3互反射模型

2.3.4互反射的定性性质

2.4一个阴影图像的形状

2.5注释

习题

编程练习

**第3章颜色**

3.1人类颜色感知

3.1.1颜色匹配

3.1.2颜色感受体

3.2颜色物理学

3.2.1颜色的来源

3.2.2表面颜色

3.3颜色表示

3.3.1线性颜色空间

3.3.2非线性颜色空间

3.4图像颜色的模型

3.4.1漫反射项

3.4.2镜面反射项

3.5基于颜色的推论

3.5.1用颜色发现镜面反射

3.5.2用颜色去除阴影

3.5.3颜色恒常性:从图像颜色获得表面颜色

3.6注释

习题

编程练习

**第二部分早期视觉:使用一幅图像**

**第4章线性滤波**

4.1线性滤波与卷积

4.1.1卷积

4.2移不变线性系统

4.2.1离散卷积

4.2.2连续卷积

4.2.3离散卷积的边缘效应

4.3空间频率和傅里叶变换

4.3.1傅里叶变换

4.4采样和混叠

4.4.1采样

4.4.2混叠

4.4.3平滑和重采样

4.5滤波器与模板

4.5.1卷积与点积

4.5.2基的改变

4.6技术:归一化相关和检测模式

4.6.1通过归一化相关检测手势的方法来控制电视机

4.7技术:尺度和图像金字塔

4.7.1高斯金字塔

4.7.2多尺度表示的应用

4.8注释

习题

编程练习

**第5章局部图像特征**

5.1计算图像梯度

5.1.1差分高斯滤波

5.2对图像梯度的表征

5.2.1基于梯度的边缘检测子

5.2.2方向

5.3查找角点和建立近邻

5.3.1查找角点

5.3.2采用尺度和方向构建近邻

5.4通过SIFT特征和HOG特征描述近邻

5.4.1SIFT特征

5.4.2HOG特征

5.5实际计算局部特征

5.6注释

习题

编程练习

**第6章纹理**

6.1利用滤波器进行局部纹理表征

6.1.1斑点和条纹

6.1.2从滤波器输出到纹理表征

6.1.3实际局部纹理表征

6.2通过纹理基元的池化纹理表征

6.2.1向量量化和纹理基元

6.2.2k均值聚类的向量量化

6.3纹理合成和对图像中的空洞进行填充

6.3.1通过局部模型采样进行合成

6.3.2填充图像中的空洞

6.4图像去噪

6.4.1非局部均值

6.4.2三维块匹配(BM3D)

6.4.3稀疏编码学习

6.4.4结果

6.5由纹理恢复形状

6.5.1在平面内由纹理恢复形状

6.5.2从弯曲表面的纹理恢复形状

6.6注释

习题

编程练习

**第三部分低层视觉:使用多幅图像**

**第7章立体视觉**

7.1双目摄像机的几何属性和对极约束

7.1.1对极几何

7.1.2本征矩阵

7.1.3基础矩阵

7.2双目重构

7.2.1图像矫正

7.3人类立体视觉

7.4双目融合的局部算法

7.4.1相关

7.4.2多尺度的边缘匹配

7.5双目融合的全局算法

7.5.1排序约束和动态规划

7.5.2平滑约束和基于图的组合优化

7.6使用多台摄像机

7.7应用:机器人导航

7.8注释

习题

编程练习

**第8章从运动中恢复三维结构**

8.1内部标定的透视摄像机

8.1.1问题的自然歧义性

8.1.2从两幅图像估计欧氏结构和运动

8.1.3从多幅图像估计欧氏结构和运动

8.2非标定的弱透视摄像机

8.2.1问题的自然歧义性

8.2.2从两幅图像恢复仿射结构和运动

8.2.3从多幅图像恢复仿射结构和运动

8.2.4从仿射到欧氏图像

8.3非标定的透视摄像机

8.3.1问题的自然歧义性

8.3.2从两幅图像恢复投影结构和运动

8.3.3从多幅图像恢复投影结构和运动

8.3.4从投影到欧氏图像

8.4注释

习题

编程练习

**第四部分中层视觉方法**

**第9章基于聚类的分割方法**

9.1人类视觉:分组和格式塔原理

9.2重要应用

9.2.1背景差分

9.2.2镜头的边界检测

9.2.3交互分割

9.2.4形成图像区域

9.3基于像素点聚类的图像分割

9.3.1基本的聚类方法

9.3.2分水岭算法

9.3.3使用k均值算法进行分割

9.3.4均值漂移:查找数据中的局部模型

9.3.5采用均值漂移进行聚类和分割

9.4分割、聚类和图论

9.4.1图论术语和相关事实

9.4.2根据图论进行凝聚式聚类

9.4.3根据图论进行分解式聚类

9.4.4归一化切割

9.5图像分割在实际中的应用

9.5.1对分割器的评估

9.6注释

习题

编程练习

**第10章分组与模型拟合**

10.1霍夫变换

10.1.1用霍夫变换拟合直线

10.1.2霍夫变换的使用

10.2拟合直线与平面

10.2.1拟合单一直线

10.2.2拟合平面

10.2.3拟合多条直线

10.3拟合曲线

10.4鲁棒性

10.4.1M估计法

10.4.2RANSAC:搜寻正常点

10.5用概率模型进行拟合

10.5.1数据缺失问题

10.5.2混合模型和隐含变量

10.5.3混合模型的EM算法

10.5.4EM算法的难点

10.6基于参数估计的运动分割

10.6.1光流和运动

10.6.2光流模型

10.6.3用分层法分割运动

10.7模型选择:哪个最好

10.7.1利用交叉验证选择模型

10.8注释

习题

编程练习

**第11章跟踪**

11.1简单跟踪策略

11.1.1基于检测的跟踪

11.1.2基于匹配的平移跟踪

11.1.3使用仿射变换来确定匹配

11.2匹配跟踪

11.2.1匹配摘要表征

11.2.2流跟踪

11.3基于卡尔曼滤波器的线性动态模型跟踪

11.3.1线性测量值和线性动态模型

11.3.2卡尔曼滤波

11.3.3前向后向平滑

11.4数据相关

11.4.1卡尔曼滤波检测方法

11.4.2数据相关的关键方法

11.5粒子滤波

11.5.1概率分布的采样表示

11.5.2最简单的粒子滤波器

11.5.3跟踪算法

11.5.4可行的粒子滤波器

11.5.5创建粒子滤波器中的粒子问题

11.6注释

习题

编程练习

**第五部分 高层视觉**

**第12章配准**

12.1刚性物体配准

12.1.1迭代最近点

12.1.2通过关联搜索转换关系

12.1.3应用:建立图像拼接

12.2基于模型的视觉:使用投影配准刚性物体

12.2.1验证:比较转换与渲染后的原图与目标图

12.3配准可形变目标

12.3.1使用主动外观模型对纹理进行变形

12.3.2实践中的主动外观模型

12.3.3应用:医疗成像系统中的配准

12.4注释

习题

编程练习

**第13章 平滑的表面及其轮廓**

13.1微分几何的元素

13.1.1曲线

13.1.2表面

13.2表面轮廓几何学

13.2.1遮挡轮廓和图形轮廓

13.2.2图像轮廓的歧点和拐点

13.2.3Koenderink定理

13.3视觉事件:微分几何的补充

13.3.1高斯映射的几何关系

13.3.2渐近曲线

13.3.3渐近球面映射

13.3.4局部视觉事件

13.3.5双切射线流形

13.3.6多重局部视觉事件

13.3.7外观图

13.4注释

习题

**第14章深度数据**

14.1主动深度传感器

14.2深度数据的分割

14.2.1分析微分几何学的基本元素

14.2.2在深度图像中寻找阶跃和顶边

14.2.3把深度图像分割为平面区域

14.3深度图像的配准和模型获取

14.3.1四元组

14.3.2使用最近点迭代方法配准深度图像

14.3.3多幅深度图像的融合

14.4物体识别

14.4.1使用解释树匹配分段平面表示的表面

14.4.2使用自旋图像匹配自由形态的曲面

14.5Kinect

14.5.1特征

14.5.2技术:决策树和随机森林

14.5.3标记像素

14.5.4计算关节位置

14.6注释

习题

编程练习

**第15章用于分类的学习**

15.1分类、误差和损失函数

15.1.1基于损失的决策

15.1.2训练误差、测试误差和过拟合

15.1.3正则化

15.1.4错误率和交叉验证

15.1.5受试者工作特征曲线(ROC)

15.2主要的分类策略

15.2.1示例:采用归一化类条件密度的马氏距离

15.2.2示例:类条件直方图和朴素贝叶斯

15.2.3示例:采用最近邻的非参分类器

15.2.4示例:线性支持向量机

15.2.5示例:核机器

15.2.6示例:级联和Adaboost

15.3构建分类器的实用方法

15.3.1手动调整训练数据并提升性能

15.3.2通过二类分类器构建多类分类器

15.3.3求解SVM和核机器的方案

15.4注释

习题

**第16章图像分类**

16.1构建好的图像特征

16.1.1示例应用

16.1.2采用GIST特征进行编码布局

16.1.3采用视觉单词总结图像

16.1.4空间金字塔

16.1.5采用主分量进行降维

16.1.6采用典型变量分析进行降维

16.1.7示例应用:检测不雅图片

16.1.8示例应用:材料分类

16.1.9示例应用:场景分类

16.2分类单一物体的图像

16.2.1图像分类策略

16.2.2图像分类的评估系统

16.2.3固定类数据集

16.2.4大量类的数据集

16.2.5花、树叶和鸟:某些特定的数据集

16.3在实践中进行图像分类

16.3.1关于图像特征的代码

16.3.2图像分类数据库

16.3.3数据库偏差

16.3.4采用众包平台进行数据库收集

16.4注释

编程练习

**第17章检测图像中的物体**

17.1滑动窗口法

17.1.1人脸检测

17.1.2行人检测

17.1.3边界检测

17.2检测形变物体

17.3物体检测算法的发展现状

17.3.1数据库和资源

17.4注释

编程练习

**第18章物体识别**

18.1物体识别应该做什么

18.1.1物体识别系统应该做什么

18.1.2目前物体识别的策略

18.1.3什么是类别

18.1.4选择:应该怎么描述

18.2特征问题

18.2.1提升当前图像特征

18.2.2其他类型的图像特征

18.3几何问题

18.4语义问题

18.4.1属性和不熟悉

18.4.2部分、姿态部件和一致性

18.4.3块的意义:部分、姿态部件、物体、短语和场景

第六部分应用与其他主题

**第19章基于图像的建模与渲染**

19.1可视外壳

19.1.1可视外壳模型的主要元素

19.1.2跟踪相交曲线

19.1.3分割相交曲线

19.1.4锥带三角化

19.1.5结果

19.1.6更进一步:雕刻可视外壳

19.2基于贴片的多视立体视觉

19.2.1PMVS模型的主要元素

19.2.2初始特征匹配

19.2.3扩张

19.2.4过滤

19.2.5结果

19.3光场

19.4注释

习题

编程练习

**第20章对人的观察**

20.1隐马尔可夫模型、动态规划和基于树形结构的模型

20.1.1隐马尔可夫模型

20.1.2关于HMM的推理

20.1.3通过EM拟合HMM

20.1.4树形结构的能量模型

20.2对图像中的人进行解析

20.2.1图形结构模型的解析

20.2.2估计衣服的表面

20.3人的跟踪

20.3.1为什么人的跟踪如此困难

20.3.2通过表面进行运动跟踪

20.3.3采用模板进行运动人体跟踪

20.4从二维到三维:提升

20.4.1在正视图进行重构

20.4.2利用外貌进行精确重构

20.4.3利用运动进行精确重构

20.5行为识别

20.5.1背景:人类运动数据

20.5.2人体结构和行为识别

20.5.3采用外貌特征识别人类行为

20.5.4采用组合的模型识别人类行为

20.6资源

20.7注释

**第21章图像搜索与检索**

21.1应用背景

21.1.1应用

21.1.2用户需求

21.1.3图像查询的类别

21.1.4什么样的用户使用图像采集

21.2源自信息检索的基本技术

21.2.1单词统计

21.2.2单词统计的平滑

21.2.3最近邻估计和哈希

21.2.4文本排序

21.3图像文件

21.3.1没有量化的匹配

21.3.2根据查询结果对图像进行排序

21.3.3浏览与布局

21.3.4图像浏览布局

21.4对注释的图片预测

21.4.1源于邻近文字的注释

21.4.2源于整幅图的注释

21.4.3采用分类器预测关联的单词

21.4.4人名与人脸

21.4.5通过分割生成标签

21.5目前最先进的单词预测器

21.5.1资源

21.5.2方法比较

21.5.3开放问题

21.6注释

**第七部分 背景材料**

**第22章优化技术**

22.1线性最小二乘法

22.1.1正则方程和伪逆

22.1.2齐次方程组和特征值问题

22.1.3广义特征值问题

22.1.4示例:拟合平面上的一条直线

22.1.5奇异值分解

22.2非线性最小二乘法

22.2.1牛顿方法:平方非线性方程组

22.2.2牛顿方法:过约束的非线性方程组

22.2.3高斯牛顿法和Levenberg-Marquardt法

22.3稀疏编码和字典学习

22.3.1稀疏编码

22.3.2字典学习

22.3.3监督字典学习

22.4最小切/最大流问题和组合优化

22.4.1最小切问题

22.4.2二次伪布尔函数

22.4.3泛化为整型变量

22.5注释

