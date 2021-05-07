

### 第9章　机器人SLAM与自主导航

[机器人SLAM与 Gmapping-Hector_slam-Cartographer--ORB_SLAM](https://www.guyuehome.com/27936)

#### 理论基础 

自主导航往往与SLAM密不可分因为SLAM生成的地图是机器人自主移动的主要蓝图。

RGB-D：红外结构光、Time-of-Flight

针对激光雷达ROS在sensor_msgs包中定义了专用数据结构——LaserScan用于存储激光消息

点云三维数据类型那么将三维数据降维到二维数据的方法也很简单即把大量数据拦腰斩断只抽取其中的一行数据重新封装为LaserScan消息
就可以获取到需要的二维激光雷达信息。这么做虽然损失了大量有效数据但是刚好可以满足2D SLAM的需求。

depthimage_to_laserscan

里程计根据传感器获取的数据来估计机器人随时间发生的位置变化。在机器人平台中较为常见的里程计是编码器

里程计根据速度对时间的积分求得位置这种方法对误差十分敏感所以采取如精确的数据采集、设备标定、数据滤波等措施是十分必要的

导航功能包要求机器人能够发布里程计nav_msgs/Odometry消息。

还包含用于滤波算法的协方差矩阵。


	 <!-- 运行joint_state_publisher节点发布机器人的关节状态-->
	 <node name="joint_state_publisher" pkg="joint_state_publisher" type="joint_state_publisher" ></node> 
	 
		<!-- 运行robot_state_publisher节点发布TF-->
	 <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher"  output="screen" >
		<param name="publish_frequency" type="double" value="50.0" />
	 </node> 

 MRobot的里程计信息在驱动节点mrobot_bringup/src/mrobot.cpp
 
 
	 // 积分计算里程计信息
	vx_ =  (vel_right.odoemtry_float + vel_left.odoemtry_float) / 2 / 1000;
	vth_ = (vel_right.odoemtry_float - vel_left.odoemtry_float) / ROBOT_LENGTH ;
	  
	curr_time = ros::Time::now();
	 
	double dt = (curr_time - last_time_).toSec();
	double delta_x = (vx_ * cos(th_) - vy_ * sin(th_)) * dt;
	double delta_y = (vx_ * sin(th_) + vy_ * cos(th_)) * dt; 
	double delta_th = vth_ * dt; 
	 
	x_ += delta_x;
	y_ += delta_y; 
	th_ += delta_th;
	last_time_ = curr_time;


发布消息： 

	msgl.child_frame_id = "base_footprint";
	msgl.twist.twist.linear.x = vx_; 
	msgl.twist.twist.linear.y = vy_; 
	msgl.twist.twist.angular.z = 

	vth_;
	msgl.twist.covariance = 

	odom_twist_covariance;
	 
	pub_.publish(msgl);


**gmapping**

[ROS进阶---ROS SLAM gmapping功能包应用方法](https://blog.csdn.net/qq_40247550/article/details/108047516)

gmapping：gmapping功能包集成了Rao-Blackwellized粒子滤波算法为开发者隐去了复杂的内部实现

gmapping功能包订阅机器人的深度信息、IMU信息和里程计信息，同时完成一些必要参数的配置，即可创建并输出基于概率的二维栅格地图。

gmapping功能包中发布/订阅的话题和服务，配置参数，坐标变换功能。

gmapping.launch： 

重点检查两个参数的输入配置：
里程计坐标系的设置odom_frame参数需要和机器人本身的里程计坐标系一致。 
激光雷达的话题名gmapping节点订阅的激光雷达话题名是“/scan”。如果与机器人发布的激光雷达话题名不一致需要使用<remap>进行重映射。

gmapping_demo.launch：

	<launch>
		<include file="$(find mrobot_navigation)/launch/gmapping.launch"/>
		<!-- 启动rviz -->
		<node pkg="rviz" type="rviz" name="rviz" args="-d $(find mrobot_navigation)/rviz/gmapping.rviz"/>
	 
	</launch>

**hector-slam**

hector_slam功能包使用高斯牛顿方法不需要里程计数据只根据激光信息便可构建地图。

hector_slam的核心节点是hector_mapping它订阅“/scan”话题以获取SLAM所需的激光数据。与gmapping相同的是hector_mapping节点也会发布map话题提供构建完成的地图信息。

不同的是hector_mapping节点还会发布slam_out_pose和poseupdate这两个话题，提供当前估计的机器人位姿。

hector-slam接口：

https://www.guyuehome.com/25572


**cartographer**

https://www.guyuehome.com/25696


cartograhper的设计目的是在计算资源有限的情况下，实时获取相对较高精度的2D地图。考虑到基于模拟策略的粒子滤波方法在较大环境下对内存和计算资源的需求较高，cartographer采用基于图网络的优化方法。

目前cartographer主要基于激光雷达来实现SLAM

**rgbdslam**

**ORB_SLAM**

基于特征点的实时单目SLAM系统，实时解算摄像机的移动轨迹，构建三维点云地图，不仅适用与手持设备获取的一组连续图像，也可以应用于汽车行驶过程中获取的连续图像


#### 导航功能包 

https://www.guyuehome.com/267

[ROS进阶---ROS中的导航框架](https://blog.csdn.net/qq_40247550/article/details/108051916)

导航的关键是机器人定位和路径规划两大部分

move_base实现机器人导航中的最优路径规划。

amcl实现二维地图中的机器人定位。

首先导航功能包需要采集机器人的传感器信息以达到实时避障的效果 。这就要求机器人通过ROS发布sensor_msgs/LaserScan或者sensor_msgs/PointCloud格式的消息也就是二维激光信息或者三维点云信息。

导航功能包要求机器人发布nav_msgs/Odometry格式的里程计信息同时也要发布相应的TF变换。

导航功能包的输出是geometry_msgs/Twist格式的控制指令这就要求机器人控制节点具备解析控制指令中线速度、角速度的能力并且最终通过这些指令控制机器人完成相应的运动。


**move_base功能包：**

全局路径规划global_planner： 使用Dijkstra或A*算法进行全局路径的规划计算出机器人到目标位置的最优路线

本地实时规划local_planner：使用Dynamic Window Approaches算法搜索躲避和行进的多条路经，综合各评价标准是否会撞击障碍物所需要的时间等，选取最优路径。

move_base功能包发布/订阅的动作、话题以及提供的服务参考列表

**amcl功能包：**

自主定位即机器人在任意状态下都可以推算出自己在地图中所处的位置

自适应或kld采样的蒙特卡罗定位方法,对已有地图使用粒子滤波器跟踪一个机器人的姿态。

amcl功能包订阅/发布的话题和提供的服务见图：


amcl定位与里程计定位对比见图：

里程计定位只是通过里程计的数据来处理/base和/odom之间的TF变换。

amcl定位可以估算机器人在地图坐标系/map下的位姿信息提供/base、/odom、/map之间的TF变换。

**代价地图的配置**

一种用于全局路径规划global_costmap，一种用于本地路径规划和实时避障local_costmap

代价地图用来存储周围环境的障碍信息

obstacle_range参数用来设置机器人检测障碍物的最大范围

raytrace_range参数用来设置机器人检测自由空间的最大范围

global_costmap_params.yaml、local_costmap_params.yaml


本地规划器base_local_planner的主要作用是根据规划的全局路径计算发布给机器人的速度控制指令。

base_local_planner_params.yaml




