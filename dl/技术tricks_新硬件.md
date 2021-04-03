
## 1 ROS相关

[创客智造 ROS入门教程](https://www.ncnynl.com/category/ros-junior-tutorial/)

[ROS学习入门（抛砖引玉篇）](https://zhuanlan.zhihu.com/p/26007106)

[ROS教程](http://wiki.ros.org/cn/ROS/Tutorials)

[机器人操作系统入门](https://www.icourse163.org/course/ISCAS-1002580008?from=searchPage)

相关资料：

机器人操作系统入门 课程讲义---Gitbook ： https://sychaichangkun.gitbooks.io/ros-tutorial-icourse163/content/chapter1/1.1.html

机器人操作系统入门 代码示例---Github ： https://github.com/DroidAITech/ROS-Academy-for-Beginners

[ROS机器人开发实践源码](https://github.com/huchunxu/ros_exploring)

ROS探索总结：  https://www.guyuehome.com/227


ROS学习网站：

[1]ROS WIKI:  wiki.ros.org 

[2]易科机器人小组(机器人、ROS学习资料)、 blog.exbot.net/

[3]古月居博客  www.guyuehome.com


## 2 ROS机器人开发实践

### 第1章　初识ROS
如何安装ROS:  

源码： https://github.com/huchunxu/ros_exploring

### 第2章　ROS架构

架构设计：三个层次OS层、中间层和应用层（Master负责管理整个系统的正常运行）

**计算图**

计算图： 节点Node就是一些执行运算任务的进程，一个系统一般由多个节点组成也可以称为“软件模块”

节点之间最重要的通信机制就是基于发布/订阅模型的消息Message通信。

TOPIC: 消息以一种发布/订阅Publish/Subscribe的方式传递。一个节点可以针对一个给定的话题Topic发布消息称为发布者Talker，也可以关注某个话题并订阅特定类型的数据称为订阅者/Listener

Service:同步传输模式为服务Service，基于客户端/服务器Client/Server模型。

ROS mater: 有一个控制器使得所有节点有条不紊地执行这就是ROS节点管理器ROS Master。ROS Master通过远程过程调用RPC提供登记列表和对其他计算图表的查找功能。

**文件系统**

	功能包Package功能包：是ROS软件中的基本单元，包含ROS节点、库、配置文件等。

	功能包清单Package Manifest：package.xml的功能包清单用于记录功能包的基本信息。

	元功能包Meta Package：组织多个用于同一目的的功能包

	消息Message类型：ROS节点之间发布/订阅的通信信息，可以使用.msg文件。

	服务Service类型：ROS客户端/服务器通信模型下的请求与应答数据类型，可以使用.srv。

**功能包结构**

	config 放置功能包中的配置文件，由用户创建文件名可以不同。

	include 放置功能包中需要用到的头文件。

	scripts 放置可以直接运行的Python脚本。

	src 放置需要编译的C++代码。

	launch 放置功能包中的所有启动文件。

	msg 放置功能包自定义的消息类型。

	srv 放置功能包自定义的服务类型。

	action 放置功能包自定义的动作指令。

	CMakeLists.txt 编译器编译功能包的规则。

	package.xml 功能包清单 

**ROS常用命令**

	catkin_create_pkg： 创建功能包

	rospack:  获取功能包信息

	catkin_make: 编译工作空间中的功能包

	rosdep: 自动安装功能包依赖的其他包

	roscd：功能包目录跳转

	roscp：拷贝功能包文件

	rosed：编辑功能包中的文件

	rosrun: 运行功能包中的可执行文件

	roslaunch: 运行启动文件

**元功能包**

元功能包是一种特殊的功能包只包含一个package.xml元功能包清单文件。它的主要作用是将多个功能包整合成为一个逻辑上独立的功能包类似于功能包集合的概念。

	<export>
	<metapackage/>
	</export> 

**ROS的通信机制**
	
	topic话题通信机制：两个节点一个是发布者Talker，另一个是订阅者Listener。两个节点分别发布、订阅同一个话题。七步建立通信。
	
	Talker注册(RPC向ROS Master注册)->Listener注册(RPC向ROS Master注册)->ROS Master进行信息匹配->Listener发送连接请求->Talker确认连接请求->Listener尝试与Talker建立网络连->Talker向Listener发布数据
	
	service服务通信机制: 五步连接法。
	
	Talker注册->Listener注册->ROS Master进行信息匹配(Master根据Listener的订阅信息从注册列表中进行查找)->Listener与Talker建立网络连接->Talker向Listener发布服务应答数据
	
	参数管理机制:类似于全局变量由ROS Master进行管理
	
	Talker设置变量(RPC)->Listener查询参数值(RPC)->ROS Master向Listener发送参数值

**话题与服务的区别**

|    |  话题 | 服务  |
|  ---- |  ---- | ----  |
| 同步性| 异步| 同步 |
| 通信模型| 发布/订阅| 客户端/服务器 |
| 底层协议| ROSTCP/ROSUDP| ROSTCP/ROSUDP |
| 反馈| 无| 有 |	
| 缓冲区| 有 | 无 |	
| 实时性| 弱 | 强  |	
| 节点关系| 多对多 | 一对多（一个server）|	
| 适用场景| 数据传输 | 逻辑处理  |		


话题是ROS中基于发布/订阅模型的异步通信模式，这种方式将信息的产生和使用双方解耦，常用于不断更新的、含有较少逻辑处理的数据通信。
而服务多用于处理ROS中的同步通信，采用客户端/服务器模型，常用于数据量较小但有强逻辑处理的数据交换。



