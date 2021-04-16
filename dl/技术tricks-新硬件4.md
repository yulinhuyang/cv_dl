
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

odom: 里程计，里程计是记录机器人与起始位置相对pose的模块，同时它还提供了机器人实时的线速度、角速度以及这些状态量的不确定性。








