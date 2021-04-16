
[自主泊车技术分析](https://zhuanlan.zhihu.com/p/135154551)

**1  ros::spin() 和 ros::spinOnce() 区别及详解**

区别在于前者调用后不会再返回，也就是你的主程序到这儿就不往下执行了，而后者在调用后还可以继续执行之后的程序。

如果你的程序写了相关的消息订阅函数，那么程序在执行过程中，除了主程序以外，ROS还会自动在后台按照你规定的格式，接受订阅的消息，但是所接到的消息并不是立刻就被处理，而是必须要等到ros::spin()或ros::spinOnce()执行的时候才被调用，这就是消息回到函数的原理

**2  ros::init() 和 ros::NodeHandle**

https://sychaichangkun.gitbooks.io/ros-tutorial-icourse163/content/chapter6/6.2.html

调用了ros::init()函数，从而初始化节点的名称和其他信息，一般我们ROS程序一开始都会以这种方式开始。创建ros::NodeHandle对象，也就是节点的句柄，它可以用来创建Publisher、Subscriber以及做其他事情。

#include<ros/ros.h>
int main(int argc, char** argv)
{
    ros::init(argc, argv, "your_node_name"); 
    ros::NodeHandle nh;
    //....节点功能
    //....
    ros::spin();//用于触发topic、service的响应队列
    return 0;
}

通常要启动节点，获取句柄，而关闭的工作系统自动帮我们完成

**3  NodeHandle常用函数**

//创建话题的publisher 
ros::Publisher advertise(const string &topic, uint32_t queue_size, bool latch=false); 

//创建话题的subscriber
ros::Subscriber subscribe(const string &topic, uint32_t queue_size, void(*)(M));

//创建服务的server，提供服务
ros::ServiceServer advertiseService(const string &service, bool(*srv_func)(Mreq &, Mres &)); 

//创建服务的client
ros::ServiceClient serviceClient(const string &service_name, bool persistent=false); 

//查询某个参数的值
bool getParam(const string &key, std::string &s); 
bool getParam (const std::string &key, double &d) const；
bool getParam (const std::string &key, int &i) const；
//从参数服务器上获取key对应的值，已重载了多个类型

//给参数赋值
void setParam (const std::string &key, const std::string &s) const；
void setParam (const std::string &key, const char *s) const;
void setParam (const std::string &key, int i) const;

**4  odom imu ego**

[odom&imu融合用于Cartographer建图](https://blog.csdn.net/zhzwang/article/details/112169035)

odom: 里程计，也就是轮速计，里程计是记录机器人与起始位置相对pose的模块，同时它还提供了机器人实时的线速度、角速度以及这些状态量的不确定性。

imu: 视觉惯性单元

cartographer中使用三种数据源去做位姿预估：odom、imu、匹配的pose。


**5 ROS 定时器**

[ROS学习定时器使用]（https://blog.csdn.net/weixin_43455581/article/details/97000848）

roscpp定时器允许用户安排一个回调发生周期性。

[ROS消息传递——std_msgs](https://blog.csdn.net/qq_36355662/article/details/62226935)

**6  quaternionMsgToTF**

[ROS-TF的使用（常用功能）](https://blog.csdn.net/liuzubing/article/details/81014240)

四元组转欧拉角：

    tf::Quaternion tf_quat;
    tf::quaternionMsgToTF(amcl_pose.pose.pose.orientation,tf_quat);
    double roll,pitch,yaw;
    tf::Matrix3x3(tf_quat).getRPY(roll,pitch,yaw);

###  7 多线程 

[C++ 参考手册](https://www.apiref.com/cpp-zh/cpp.html)

[c++之多线程中“锁”的基本用法](https://zhuanlan.zhihu.com/p/91062516)

[c++11 多线程（3）atomic 总结](https://www.jianshu.com/p/8c1bb012d5f8)

atomic_load： 在原子对象中原子性地获取存储的值(函数模板)

**condition_variable**

std::condition_variable 类是同步原语，能用于阻塞一个线程，或同时阻塞多个线程，直至另一线程修改共享变量（条件）并通知 condition_variable 。

有意修改变量的线程必须:

    获得 std::mutex （常通过 std::lock_guard ）
    在保有锁时进行修改
    在 std::condition_variable 上执行 notify_one 或 notify_all （不需要为通知保有锁）

即使共享变量是原子的，也必须在互斥下修改它，以正确地发布修改到等待的线程。

任何有意在 std::condition_variable 上等待的线程必须：

    在与用于保护共享变量者相同的互斥上获得 std::unique_lock<std::mutex>

    执行下列之一：

        检查条件，是否为已更新或提醒它的情况
        执行 wait 、 wait_for 或 wait_until ，等待操作自动释放互斥，并悬挂线程的执行
        condition_variable 被通知时，时限消失或虚假唤醒发生，线程被唤醒，且自动重获得互斥。之后线程应检查条件，若唤醒是虚假的，则继续等待。

### 8 EKF 

[卡尔曼滤波（3）-- EKF, UKF](https://zhuanlan.zhihu.com/p/59681380)









