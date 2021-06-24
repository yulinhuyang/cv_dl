python ... 

	a[:, :, None]和a[…, None]的输出是一样的，就是因为…代替了前面两个冒号
	
	
深度卷积K*K*1 的输出channel是1，分离卷积1*1*K 的kernel是1




MASK RCNN:

	RPN + ROI-Align + Fast-RCNN + FCN


	
	
Yolov4  tiny 模型： 23.1MB,  40.2% AP50, 371 FPS   <----------------->   可以对比：yolo  V5s: 27MB，400 FPS

分辨率： 416 x  416  608 x 608 


Yolo.py:
detect_image :   letterbox_image  --> net --> yolo_decodes ---> non_max_suppression---> yolo_correct_boxes


剪枝减什么：根据权重减去channel。

center loss 原理：

柱面投影原理： 相机标定、相关搞清楚

抽帧的功耗问题：命令模式说清楚，状态机。
	

### 1  

	LR: 分类算法，判别式模型。logistic loss  log(1+e-x) ，考虑所有点

	SVM:分类算法，判别式模型。hinge loss函数  max(0,1-z)，只考虑边界点，非线性问题靠核函数。需要先归一化，自带正则化项 1/2 w*w。

	若feature数远大于样本数量，使用LR算法或者Linear SVM

	若feature数较小，样本数量适中，使用kernel SVM

	若feature数较小，样本数很大，增加更多的feature然后使用LR算法或者Linear SVM
	

	boosting调整的两个权重，一个样本权重，一个是loss权重

	

 DeepSort中最大的特点是加入外观信息，借用了ReID领域模型来提取特征，减少了ID switch的次数
 
 马氏距离(Mahalanobis Distance)，应对高维线性分布的数据中各维度间非独立同分布的问题。 
 
 Cost Matrix: 外观模型Reid + 运动模型(马氏距离)

### 2  C++

引用与指针： 指针是实体，要初始化,间接访问，传参是传递指针的地址， 引用是别名，直接访问。传引用是传递变量的地址。

传值与传引用：传值是传递参数的拷贝，传引用是实参变量的地址。

static:

	全局可见性，分配一次。不能被virtual修饰。修饰的变量要类外初始化，修饰的函数不能访问非static的类成员。

	static变量，初始化一次，多次赋值。主程序之前内存分配。

const:阻止变量被修改，定义的时候就初始化。类型转换const_cast转为非const类型。

深复制：开辟新内存地址存放复制的对象。浅复制：只指向被复制的内存地址。

class与struct: class private ，strcut public,

虚函数与inline: 虚函数，实现运行时多态; inline,小函数，编译期间替换函数代码。

列表初始化：快于赋值初始化


构造函数执行顺序：基类-->成员类-->派生类

析构函数：派生类-->成员类-->基类

类的关系：has-A  包含,use-A 组合,is-A 继承


C语言执行过程： 源代码－－>预处理－－>编译（.s）－－>优化－－>汇编（.o）－－>链接-->可执行文件


堆：程序员管理，申请或释放，向上生长，动态分配。堆大小4G。

栈：系统管理，向下生长。栈1M大小。



