
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

### 第3章　ROS基础

**第一个ROS例程——小乌龟仿真**

turtlesim

话题订阅、发布 

服务

参数


**创建工作空间和功能包**

src代码空间： Source Space开发过程中最常用的文件夹用来存储所有ROS功能包的源码文件。

build编译空间： Build Space用来存储工作空间编译过程中产生的缓存信息和中间文件。

devel开发空间： Development Space用来放置编译生成的可执行文件。

install安装空间： Install Space编译成功后可以使用make install命令将可执行文件安装到该空间

创建工作空间: 

	$ mkdir -p ~/catkin_ws/src
	$ cd ~/catkin_ws/src 
	$ catkin_init_workspace

	$ cd ~/catkin_ws/
	$ catkin_make 
	
	echo"source/WORKSPACE/devel/setup.bash">>~/.bashrc
 
直接创建功能包的命令catkin_create_pkg

	$ catkin_create_pkg <package_name> [depend1] [depend2] [depend3]
	
然后回到工作空间的根目录下进行编译并且设置环境变量

	$ cd ~/catkin_ws
	$ catkin_make 
	$ source 

	~/catkin_ws/devel/setup.bash


RoboWare简介

**话题中的Publisher与Subscriber**

Publisher的主要作用是针对指定话题发布特定数据类型的消息

	#include <sstream>
	#include "ros/ros.h"
	#include "std_msgs/String.h"
	ros::init(argc,  argv, "talker");
	// 创建节点句柄
	ros::NodeHandle n;
	// 创建一个Publisher发布名为chatter的topic消息类型为std_msgs::String
	ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1000);
    
	// 发布消息
	ROS_INFO("%s", msg.data.c_str());
	chatter_pub.publish(msg); 
	// 循环等待回调函数
	ros::spinOnce(); 
 
创建Subscriber

	#include "ros/ros.h"
	#include "std_msgs/String.h"

	// 接收到订阅的消息后会进入消息回调函数
	void chatterCallback(const std_msgs::String::ConstPtr& msg)
	{ 
		// 将接收到的消息打印出来
		ROS_INFO("I heard: [%s]", msg->data.c_str());
	} 

	int main(int argc, char **argv)
	{ 
		// 初始化ROS节点
		ros::init(argc,  argv, "listener");

		// 创建节点句柄
		ros::NodeHandle n;

		// 创建一个Subscriber订阅名为chatter的话题注册回调函数chatterCallback
		ros::Subscriber sub = n.subscribe("chatter", 1000, chatterCallback);
		
		// 循环等待回调函数
		ros::spin(); 
		return 0; 
	} 
 
编译功能包配置项：

include_directories 设置头文件的相对路径

add_executable  设置需要编译的代码和生成的可执行文件

target_link_libraries   设置链接库

add_dependencies   设置依赖

自定义msg

ROS的元功能包common_msgs中提供了许多不同消息类型的功能包，如std_msgs标准数据类型、geometry_msgs几何学数据类型、sensor_msgs传感器数据类型等

**服务中的Server和Client**

自定义服务数据：/srv/AddTwoInts.srv

	int64 a
	int64 b 
	--- 
	 
	int64 sum

并在package.xml 和cmake 里面配置依赖

创建Server；

	#include "ros/ros.h"
	#include "learning_communication/AddTwoInts.h"

	// service回调函数输入参数req输出参数res
	bool add(learning_communication::AddTwoInts::Request  &req,
			 learning_communication::AddTwoInts::Response &res) 
	{ 
	 // 将输入参数中的请求数据相加结果放到应答变量中
	 res.sum = req.a + req.b; 
	 ROS_INFO("request: x=%ld, y=%ld", (long int)req.a, (long int)req.b);
	 ROS_INFO("sending back response: [%ld]", (long int)res.sum); 
	 return true; 
	} 
	 
	int main(int argc, char **argv)
	{ 
		// ROS节点初始化
		ros::init(argc,  argv, "add_two_ints_server");
		
		// 创建节点句柄
		ros::NodeHandle n;

		// 创建一个名为add_two_ints的server注册回调函数add()
		ros::ServiceServer service = n.advertiseService("add_two_ints", add);

		// 循环等待回调函数
		ROS_INFO("Ready  to add two ints.");
		ros::spin(); 
		return 0; 
	} 
 
创建Client：

	#include <cstdlib>
	#include "ros/ros.h"
	#include "learning_communication/AddTwoInts.h"
	int main(int argc, char **argv) 
	{ 
		// ROS节点初始化
		ros::init(argc,  argv, "add_two_ints_client");

		// 从终端命令行获取两个加数
		if (argc != 3) 
		{ 
			ROS_INFO("usage: add_two_ints_client X Y");
			return 1; 
		} 
	 
		// 创建节点句柄
		ros::NodeHandle n;

		// 创建一个client请求add_two_int service

		// service消息类型是learning_communication::AddTwoInts
		ros::ServiceClient client = n.serviceClient<learning_communication::AddTwoInts> ("add_two_ints");

		// 创建learning_communication::AddTwoInts类型的service消息
		learning_communication::AddTwoInts srv; 
		srv.request.a = atoll(argv[1]); 
		srv.request.b = atoll(argv[2]); 
		 
		// 发布service请求等待加法运算的应答结果
		if (client.call(srv)) 
		{ 
			 ROS_INFO("Sum: %ld", (long int)srv.response.sum);
		} 
		else
		{ 
			ROS_ERROR("Failed to call service add_two_ints");
			return 1; 
		} 

		return 0;
	} 

**ROS中的命名空间**

	ROS中的节点、参数、话题和服务统称为计算图源。
	
	基础base名称例如base。

	全局global名称例如/global/name。

	相对relative名称例如relative/name，默认。

	私有private名称例如~private/name。
	
	ROS的命名解析是在命名重映射之前发生的。
	
	name:=new_name

分布式多机通信：ROS中只允许存在一个Master，在多机系统中Master只能运行在一台机器上其他机器需要通过ssh的方式和Master取得联系

### 第4章　ROS中的常用组件

**launch启动文件**
launch启动文件通过XML文件实现多节点的配置和启动。

	<launch>
		<node pkg="turtlesim" name="sim1" type="turtlesim_node"/>
		<node pkg="turtlesim" name="sim2" type="turtlesim_node"/> 
	</launch> 

	标签元素有两个<param>和<arg>，一个代表parameter，另一个代表argument
	
	<param name="output_frame" value="odom"/>
	
	<arg name="arg-name" default= "arg-value"/>
 
	重映射机制：  <remap from="/turtlebot/cmd_vel" to="/cmd_vel"/>
	
	嵌套复用： <include file="$(dirname)/other.launch" />
	
	
**TF坐标变换**

TF坐标变换管理机器人系统中繁杂的坐标系变换关系。

监听TF变换： 接收并缓存系统中发布的所有坐标变换数据并从中查询所需要的坐标变换关系。

广播TF变换：向系统中广播坐标系之间的坐标变换关系。系统中可能会存在多个不同部分的TF变换广播，每个广播都可以直接将坐标变换关系插入TF树中不需要再进行同步。

tf_monitor： 打印TF树中所有坐标系的发布状态， 
	
	tf_monitor <source_frame> <target_target>

tf_echo：查看指定坐标系之间的变换关系
	
	tf_echo <source_frame> <target_frame>
	
static_transform_publisher：发布两个坐标系之间的静态坐标变换，这两个坐标系不发生相对位置变化	
	
	$ static_transform_publisher x y z yaw pitch roll frame_id child_frame_id  period_in_ms
	$ static_transform_publisher x y z qx qy qz qw frame_id child_frame_id   period_in_ms

view_frames:生成pdf文件显示整棵TF树的信息

	rosrun tf view_frames
	
创建TF广播器:

	std::string turtle_name;

	void poseCallback(const turtlesim::PoseConstPtr& msg)
	{
		// tf广播器
		static tf::TransformBroadcaster br;

		// 根据乌龟当前的位姿，设置相对于世界坐标系的坐标变换
		tf::Transform transform;
		transform.setOrigin( tf::Vector3(msg->x, msg->y, 0.0) );
		tf::Quaternion q;
		q.setRPY(0, 0, msg->theta);
		transform.setRotation(q);

		// 发布坐标变换
		br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", turtle_name));
	}

创建TF监听器：

	tf::StampedTransform transform;
	try
	{
		// 查找turtle2与turtle1的坐标变换
		listener.waitForTransform("/turtle2", "/turtle1", ros::Time(0), ros::Duration(3.0));
		listener.lookupTransform("/turtle2", "/turtle1", ros::Time(0), transform);
	}
	catch (tf::TransformException &ex) 
	{
		ROS_ERROR("%s",ex.what());
		ros::Duration(1.0).sleep();
		continue;
	}

	// 根据turtle1和turtle2之间的坐标变换，计算turtle2需要运动的线速度和角速度
	// 并发布速度控制指令，使turtle2向turtle1移动
	geometry_msgs::Twist vel_msg;
	vel_msg.angular.z = 4.0 * atan2(transform.getOrigin().y(),
									transform.getOrigin().x());
	vel_msg.linear.x = 0.5 * sqrt(pow(transform.getOrigin().x(), 2) +
								  pow(transform.getOrigin().y(), 2));
	turtle_vel.publish(vel_msg);
	

接口： waitForTransform、lookupTransform


**Qt工具箱**
Qt工具箱提供多种机器人开发的可视化工具如日志输出、计算图可视化、数据绘图、参数动态配置等功能。

相关命令：

rqt_console工具用来图像化显示和过滤ROS系统运行状态中的所有日志消息

rqt_graph 图形化显示当前ROS系统中的计算图

rqt_plot 二维数值曲线绘制工具

rqt_reconfigure 动态配置ROS系统中的参数

**rviz三维**

rviz三维可视化平台实现机器人开发过程中多种数据的可视化显示并且可通过插件机制无限扩展。


**gazebo仿真**

gazebo仿真环境创建仿真环境并实现带有物理属性的机器人仿真。

Building Editor工具可以手动绘制地图

**rosbag数据记录**

rosbag数据记录与回放记录并回放ROS系统中运行时的所有话题信息方便后期调试使用。

	抓取消息
	
	$ mkdir ~/bagfiles
	$ cd ~/bagfiles 
	$ rosbag record  -a
	
	回放消息
	
	rosbag info <your bagfile>
	
	rosbag play <your bagfile>



### 第5章　机器人平台搭建

组成部分：执行机构、驱动系统、传感系统和控制系统

MRobot是ROSClub基于ROS系统构建的一款差分轮式移动机器人

usb_cam：针对V4L协议USB摄像头的ROS驱动包核心节点


基于Raspberry Pi的控制系统实现

为机器人装配摄像头

为机器人装配Kinect

为机器人装配激光雷达

激光雷达可用于测量机器人和其他物体之间的距离

### 第6章　机器人建模与仿真

**统一机器人描述格式——URDF**

<link>标签：描述机器人某个刚体部分的外观和物理属性包括尺寸size、颜色color、形状shape、惯性矩阵inertial matrix、碰撞参数collision properties等

	<link name="<link name>">
	<inertial> . . . . . .</inertial>
		<visual> . . . . . . </visual> 
	    <collision> . . . . . . </collision>
	</link> 

	<visual>标签用于描述机器人link部分的外观参数，<inertial>标签用于描述link的惯性参数，而<collision>标签用于描述link的碰撞属性	

<joint>标签：描述机器人关节的运动学和动力学属性，包括关节运动的位置和速度限制
	
	连接两个刚体link这两个link分别称为parent link和child link
	
<robot>：完整机器人模型的最顶层标签，<link>和<joint>标签都必须包含在<robot>标签内。
	
	<robot name="<name of the robot>">
		<link> ....... </link> 
		<link> ....... </link> 
		<joint> ....... </joint>
		<joint> ....... </joint> 
	</robot> 
 
<gazebo>：描述机器人模型在Gazebo中仿真所需要的参数
	
	<gazebo reference="link_1">
		<material>Gazebo/Black</material>
	</gazebo> 
 
	
**创建机器人URDF模型**

catkin_create_pkg mrobot_description urdf xacro

check_urdf：对mrobot_chassis.urdf文件进行检查

urdf_to_graphiz： 查看URDF模型的整体结构

**改进URDF模型**

在base_link中加入<inertial>和<collision>标签描述机器人的物理惯性属性和碰撞属性

惯性参数的设置主要包含质量和惯性矩阵

使用xacro优化URDF：xacro是一个精简版本的URDF文件。xacro是URDF的升级版模型文件的后缀名由.urdf变为.xacro。

直接在启动文件中调用xacro解析器，自动将xacro转换成URDF文件。

**添加传感器模型**

顶层xacro文件把机器人和摄像头这两个模块拼装：

<?xml version="1.0"?>
<robot name="mrobot"  xmlns:xacro="http://www.ros.org/wiki/xacro">
 <xacro:include filename="$(find mrobot_description)/urdf/mrobot_body.urdf.xacro" />
 <xacro:include filename="$(find mrobot_description)/urdf/camera.xacro"/>


基于ArbotiX和rviz的仿真器

**ros_control**

https://www.guyuehome.com/890

ros_control就是ROS为用户提供的应用与机器人之间的中间件，包含一系列控制器接口、传动装置接口、硬件接口、控制器工具箱等等。

硬件抽象层负责机器人硬件资源的管理，而controller从抽象层请求资源即可，并不直接接触硬件.

Controller Manager、Controller、Hardware Rescource、RobotHW、Real Robot

Gazebo仿真



### 第7章　机器视觉

ROS中的图像数据：二维图像数据、三维点云数据

**摄像头标定**

使用camera_calibration功能包试下双目和单目摄像头的标定,标定kinect的RGB和红外摄像头

**OpenCV库**

ROS中已经集成了opencv库和相关的接口包ros-kinetic ， cv_bridge 提供了opencv和ROS图像数据转换的接口

一个ROS节点订阅摄像头驱动发布的图像消息，然后转换未Opencv的图像格式进行显示，然后再转换未ROS 图像消息

imgmsg_to_cv2()：将ROS图像消息转换成OpenCV图像数据，该接口有两个输入参数第一个参数指向图像消息流，第二个参数用来定义转换的图像数据格式。

cv2_to_imgmsg()：将OpenCV格式的图像数据转换成ROS图像消息，该接口同样要求输入图像数据流和数据格式这两个参数。

OpenCV中的人脸识别算法首先将获取的图像进行灰度化转换，并进行边缘处理与噪声过滤然后将图像缩小、直方图均衡化，同时将匹配分类器放大相同倍数，直到匹配分类器的大小大于检测图像，则返回匹配结果。

匹配过程中可以根据cascade分类器中的不同类型分别进行匹配例如正脸和侧脸。


# 创建cv_bridge
    self.bridge = CvBridge()
	
    self.image_pub = rospy.Publisher("cv_bridge_image", Image, queue_size=1)
	
	self.image_sub = rospy.Subscriber("input_rgb_image", Image,self.image_callback, queue_size=1)
	

物体跟踪

首先根据输入的图像流和选择跟踪的物体采样物体在图像当前帧中的特征点，然后将当前帧和下一帧图像进行灰度值比较，估计出当前帧中跟踪物体的特征点在下一帧图像中的位置。
再过滤位置不变的特征点，余下的点就是跟踪物体在第二帧图像中的特征点其特征点集群即为跟踪物体的位置。

二维码识别

ar_track_alvar允许创建多种二维码标签

sudo apt-get install ros-kinetic-ar-track-alvar

Object Recognition KitchenORK其中包含了多种三维物体识别的方法

ORK中的大部分算法思路都是模板匹配也就是说首先建立已知物体的数据模型然后根据采集到的场景信息逐一进行匹配找到与数据库中匹配的物体即可确定识别到的物体

couchdb工具

