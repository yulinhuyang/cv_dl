
主要参考：

https://www.bilibili.com/video/BV1qV41167d2?p=2

http://www.cvlibs.net/datasets/kitti/

https://bdd-data.berkeley.edu/


####1.  v1 发布camera 和img
  
    kitti数据下载链接： 

	http://www.cvlibs.net/datasets/kitti/raw_data.php

	https://github.com/tomas789/kitti2bag

[KITTI数据集--参数](https://blog.csdn.net/cuichuanchen3307/article/details/80596689)

点云的格式： x  y  z + Intensity(激光反射强度)

truncated：是否被截断

occluded：是否可见

alpha：物体的观察角度，[-pi,pi]


16个标注的格式： frame + track id  + type + truncated + occluded + alpha + bbox(left，top,right，bottom)+ dimensions(height,width,length) + location(x,y,z) + rotation_y + score(confidence)  
 
发布过程：初始化节点 --->读取数据 --->发布出去

发布照片和点云：

``` python

    frame = 0
    rospy.init_node('kitti_node',anonymous=True)
    cam_pub = rospy.Publisher('kitti_cam', Image, queue_size=10)
    pcl_pub = rospy.Publisher('kitti_point_cloud', PointCloud2, queue_size=10)
    bridge = CvBridge()

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        img = cv2.imread(os.path.join(DATA_PATH, 'image_02/data/%010d.png'%frame))
        cam_pub.publish(bridge.cv2_to_imgmsg(img,"bgr8"))

        point_cloud = np.fromfile(os.path.join(DATA_PATH, 'velodyne_points/data/%010d.bin'%frame),dtype=np.float32).reshape(-1,4)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        pcl_pub.publish(pcl2.create_cloud_xyz32(header, point_cloud[:,:3]))
        rospy.loginfo("camera image published")
        rate.sleep()
        frame += 1
        frame %= 154
``` 


####2. V2  小函数拆分，发布imu和车模型等其他传感器

https://github.com/seaside2mm/ros-kitti-project/tree/master/src/v2-publish-other-sensor


data_utils.py

```python

```


kitti-1.py
```python


```

publish_utils.py
``` python


``` 



注意：

1 时间戳用法

``` python

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = FRAME_ID
``` 
	
2 GPS格式

```python
    gps = NavSatFix()
    gps.header.frame_id = FRAME_ID
    gps.header.stamp = rospy.Time.now()
    gps.latitude = gps_data.lat
    gps.longitude = gps_data.lon
    gps.altitude = gps_data.alt
```
3 imu格式

```python

Publish IMU data
http://docs.ros.org/melodic/api/sensor_msgs/html/msg/Imu.html

```

    imu = Imu()
    imu.header.frame_id = FRAME_ID
    imu.header.stamp = rospy.Time.now()
    q = tf.transformations.quaternion_from_euler(float(imu_data.roll), float(imu_data.pitch), \
                                                     float(imu_data.yaw)) # prevent the data from being overwritten
    imu.orientation.x = q[0]
    imu.orientation.y = q[1]
    imu.orientation.z = q[2]
    imu.orientation.w = q[3]
    imu.linear_acceleration.x = imu_data.af
    imu.linear_acceleration.y = imu_data.al
    imu.linear_acceleration.z = imu_data.au
    imu.angular_velocity.x = imu_data.wf
    imu.angular_velocity.y = imu_data.wl
    imu.angular_velocity.z = imu_data.wu

    imu_pub.publish(imu)

	
4  Marker用法

```python
    marker = Marker()
    marker.header.frame_id = FRAME_ID
    marker.header.stamp = rospy.Time.now()

    marker.id = 0
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration()
    marker.type = Marker.LINE_STRIP
    # line
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.scale.x = 0.2 # line width

    marker.points = []

    # check the kitti axis model 
    marker.points.append(Point(5,-5,0)) # left up
    marker.points.append(Point(0,0,0)) # center
	marker.points.append(Point(5, 5,0)) # right up
```
	
5   发布dae的car模型：

```python
    mesh_marker.id = -1
    mesh_marker.lifetime = rospy.Duration()
    mesh_marker.type = Marker.MESH_RESOURCE
    mesh_marker.mesh_resource = "/root/catkin_ws/src/kitti_tutorial/AudiR8.dae"
	
	mesh_marker.pose.position.x = 0.0
    mesh_marker.pose.position.y = 0.0
    mesh_marker.pose.position.z = -1.73

    q = tf.transformations.quaternion_from_euler(np.pi/2,0,np.pi)
    mesh_marker.pose.orientation.x = q[0]
    mesh_marker.pose.orientation.y = q[1]
    mesh_marker.pose.orientation.z = q[2]
    mesh_marker.pose.orientation.w = q[3]
	
	mesh_marker.color.r = 1.0
    mesh_marker.color.g = 1.0
    mesh_marker.color.b = 1.0
    mesh_marker.color.a = 1.0

```	


#### 3.  V3  tricking 发布3d box

[解决Github加载ipynb文件缓慢/失败](https://blog.csdn.net/weiwei9363/article/details/79438908)

格式： https://github.com/pratikac/kitti/blob/master/readme.tracking.txt

    alpha, xmin，ymin，xmax，ymax，high，width，long，rotation_y，置信度score

下载链接： https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip

marker: http://wiki.ros.org/rviz/DisplayTypes/Marker  Line List

MarkerArray


8个顶点：


```python

juper notebook:

2d框显示，迭代所有：

	COLOR_DICT = {'Car':(255,255,0), 'Cyclist':(255,0,255), 'Pedestrian':(255,55,25)}

	for typ, box in zip(types,boxes):
	  top_left = int(box[0]),int(box[1])
	  bottom_right = int(box[2]),int(box[3])
	  cv2.rectangle(image, top_left, bottom_right,COLOR_DICT[typ],2)
```

publish_utils.py：

发布2D框：

```python

def publish_camera(cam_pub, bridge, image, borders_2d_cam2s=None, object_types=None, log=False)

	if borders_2d_cam2s is not None:
		for i, box in enumerate(borders_2d_cam2s):
			top_left = int(box[0]), int(box[1])
			bottom_right = int(box[2]), int(box[3])
			if object_types is None:
				cv2.rectangle(image, top_left, bottom_right, (255,255,0), 2)
			else:
				cv2.rectangle(image, top_left, bottom_right, DETECTION_COLOR_MAP[object_types[i]], 2) 
```	
	
发布点云：

```python
from mpl_toolkits.mplot3d import Axes3D	
	
def draw_point_cloud(ax, points, axes=[0, 1, 2], point_size = 0.1, xlim3d=None, ylim3d=None, zlim3d=None):

	axes_limits = [
	   [-20,80],
	   [-20,20],
	   [-3,3]
	]
	axes_str = ['X','Y','Z']
	ax.grid = False
```
	
发布3D框：

    x,y,z,w,h,l  ---> 计算出来8个点

	center在box的下侧平面中心
```python
	x_corners = [l/2, l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]^T
	y_corners = [0,   0,    0,    0,   -h,   -h,   -h,   -h  ]^T
	z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2  ]^T

	def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
		"""
		Return : 3xn in cam2 coordinate
		"""
		R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
		x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
		y_corners = [0,0,0,0,-h,-h,-h,-h]
		z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
		corners_3d_cam2 = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
		corners_3d_cam2[0,:] += x
		corners_3d_cam2[1,:] += y
		corners_3d_cam2[2,:] += z
		return corners_3d_cam2
		
	# 在velodyne坐标系中，发布3d boxes
	LINES = [[0, 1], [1, 2], [2, 3], [3, 0]] # lower face
	LINES+= [[4, 5], [5, 6], [6, 7], [7, 4]] # upper face
	LINES+= [[4, 0], [5, 1], [6, 2], [7, 3]] # connect lower face and upper face
	LINES+= [[4, 1], [5, 0]] # front face and draw x
		
	def publish_3dbox(box3d_pub, corners_3d_velos, track_ids, types=None, publish_id=True):
		
		for i, corners_3d_velo in enumerate(corners_3d_velos):
			# 3d box
			marker = Marker()
			marker.header.frame_id = FRAME_ID
			marker.header.stamp = rospy.Time.now()
			marker.id = i
			marker.action = Marker.ADD
			marker.lifetime = rospy.Duration(LIFETIME)
			marker.type = Marker.LINE_LIST
		
			b, g, r = DETECTION_COLOR_MAP[types[i]]
			
			for l in LINES:
				p1 = corners_3d_velo[l[0]]
				marker.points.append(Point(p1[0], p1[1], p1[2]))
				p2 = corners_3d_velo[l[1]]
				marker.points.append(Point(p2[0], p2[1], p2[2]))
				marker_array.markers.append(marker)
			
			...
			
			# add track id
			track_id = track_ids[i]
            text_marker = Marker()
            text_marker.header.frame_id = FRAME_ID
            text_marker.header.stamp = rospy.Time.now()

            text_marker.id = track_id + 1000
            text_marker.action = Marker.ADD
            text_marker.lifetime = rospy.Duration(LIFETIME)
            text_marker.type = Marker.TEXT_VIEW_FACING

            p4 = corners_3d_velo[4] # upper front left corner

            text_marker.pose.position.x = p4[0]
            text_marker.pose.position.y = p4[1]
            text_marker.pose.position.z = p4[2] + 0.5
			
			text_marker.text = str(track_id)
			
			marker_array.markers.append(text_marker)
			
		
	box3d_pub = rospy.Publisher('kitti_3dbox',MarkerArray, queue_size=10) --> publish_3dbox(box3d_pub, corner_3d_velos,  track_ids, types)
```	
	

Calibration坐标系转换：
	
	8个点的3d检测框---> 投影到velodyne坐标系

	定义位置：
	https://github.com/charlesq34/frustum-pointnets/blob/master/kitti/kitti_util.py

```python

	class Calibration(object):
	
	#读取kitti标定文件
	calib = Calibration('utils',from_video=True)
	# 从2号相机坐标系-->投影到激光雷达坐标系
	boxes_3d = np.array(df_tracking_frame[['height','width','length','pos_x','pos_y','pos_z','rot_y']])
	corner_3d_velos = []

	for box_3d in boxes_3d:
		corner_3d_cam2 = compute_3d_box_cam2(*box_3d)
		corner_3d_velo = calib.project_rect_to_velo(np.array(corner_3d_cam2).T)
		corner_3d_velos += [corner_3d_velo]    # += 类似append，不完全相同	
```


	
#### v4  plot imu-odom
	
绘图软件：https://sketch.io/sketchpad/

	
##### 读取IMU GPS信息并计算距离以及选择角度

IMU/GPS信息：https://github.com/pratikac/kitti/blob/master/readme.raw.txt

- lat:     latitude of the oxts-unit (deg)
- lon:     longitude of the oxts-unit (deg)
- alt:     altitude of the oxts-unit (m)
- roll:    roll angle (rad),  0 = level, positive = left side up (-pi..pi)
- pitch:   pitch angle (rad), 0 = level, positive = front down (-pi/2..pi/2)
- yaw:     heading (rad),     0 = east,  positive = counter clockwise (-pi..pi)
- vn:      velocity towards north (m/s)
- ve:      velocity towards east (m/s)
- vf:      forward velocity, i.e. parallel to earth-surface (m/s)
- vl:      leftward velocity, i.e. parallel to earth-surface (m/s)


通过imu的yaw变化，计算两帧之间的旋转角度
	
	yaw_change = float(imu_data.yaw - prev_imu_data.yaw)

做大圆，计算两点间的距离：https://en.wikipedia.org/wiki/Great-circle_distance

```python

	#1 利用gps计算两帧距离，大圆距离计算法
	def compute_great_circle_distance(lat1, lon1, lat2, lon2):
		"""
		Compute the great circle distance from two gps data
		Input   : latitudes and longitudes in degree
		Output  : distance in meter
		"""
		delta_sigma = float(np.sin(lat1*np.pi/180)*np.sin(lat2*np.pi/180)+ \
							np.cos(lat1*np.pi/180)*np.cos(lat2*np.pi/180)*np.cos(lon1*np.pi/180-lon2*np.pi/180))
		if np.abs(delta_sigma) > 1:
			return 0.0
		return 6371000.0 * np.arccos(delta_sigma)

	# 计算所有
	def compute_every_two_frame_distance():
	  prev_imu_data = None
	  gps_distances = []
	  imu_distances = []
	  for frame in range(130):
		imu_data = read_imu('%010d.txt'%frame)

		if prev_imu_data is not None:
		  gps_distances += [compute_great_circle_distance(imu_data.lat, imu_data.lon, prev_imu_data.lat, prev_imu_data.lon)]
		  # 速度*时间，累加法
		  imu_distances += [0.1*np.linalg.norm(imu_data[['vf','vl']])] # 0.1s移动距离
		prev_imu_data = imu_data
	  return gps_distances, imu_distances

	gps_distances,imu_distances = compute_every_two_frame_distance()
	plt.figure()
	plt.plot(gps_distances, label='gps')
	plt.plot(imu_distances, label='imu')
		
	imu短时间误差很小，但是长时间误差累加，就不如GPS了
```	
	
##### 画出自己的轨迹

得到当前这一帧时，所有过去位置的坐标

```python
def compute_locations():
  prev_imu_data = None
  locations = []
  for frame in range(130):
    imu_data = read_imu('%010d.txt'%frame)

    if prev_imu_data is not None:
      displacement = [0.1*np.linalg.norm(imu_data[['vf','vl']])] # 0.1s移动距离
      yaw_change = float(imu_data.yaw - prev_imu_data.yaw)
      for i in range(len(locations)):
        x0, y0 = locations[i]
        # 前一帧坐标变换到现在帧坐标
        x1 = x0*np.cos(yaw_change) + y0*np.sin(yaw_change) - displacement
        y1 = -x0*np.sin(yaw_change) + y0*np.cos(yaw_change)
        locations[i] = [x1,y1]

    locations += [[0,0]] #相对自己
    prev_imu_data = imu_data
```


##### 画出自己的轨迹 ---> 移植到v4 project中实现

https://github.com/seaside2mm/ros-kitti-project/blob/master/src/v4-plot-imu-odom/utils.py


```python

utils.py，只保留最多max_length个数量的存储的最近的轨迹信息

class Object():
    #trajectory
    def __init__(self, center, max_length):
        self.locations = deque(maxlen=max_length) # save loc
        self.locations.appendleft(center)
        self.max_length = max_length

    def update(self, center, displacement, yaw):
        """
        Update the center of the object, and calculates the velocity
        """
        for i in range(len(self.locations)):
            x0, y0 = self.locations[i]
            x1 = x0 * np.cos(yaw) + y0 * np.sin(yaw) - displacement
            y1 = -x0 * np.sin(yaw) + y0 * np.cos(yaw)
            self.locations[i] = np.array([x1, y1])

        if center is not None:
            self.locations.appendleft(center) 
	
	def reset(self):
        self.locations = deque(maxlen=self.max_length)
```

publish_ego_car 函数

更新轨迹的逻辑:

```python

https://github.com/seaside2mm/ros-kitti-project/blob/master/src/v4-plot-imu-odom/kitti.py

corner_3d_velos = []
centers = {} # current frame tracker. track id:center
for track_id, box_3d in zip(track_ids, boxes_3d):
	corner_3d_cam2 = compute_3d_box_cam2(*box_3d)
	corner_3d_velo = calib.project_rect_to_velo(np.array(corner_3d_cam2).T)
	corner_3d_velos += [corner_3d_velo] # one bbox 8 x 3 array
	centers[track_id] = np.mean(corner_3d_velo, axis=0)[:2] # get ccenter of every bbox, don't care about height

centers[-1] = np.array([0,0]) # for ego car, we set its id = -1, center[0,0]，加入物体每一帧的当前中心

centers：当前帧的物体

tracker：所有追踪的物体，当前帧可能没有

if prev_imu_data is None:
	for track_id in centers:
		tracker[track_id] = Object(centers[track_id], 20)
else:
	displacement = 0.1*np.linalg.norm(imu_data[['vf','vl']])
	yaw_change = float(imu_data.yaw - prev_imu_data.yaw)
	print track_id
	for track_id in centers: # for one frame id 
		if track_id in tracker:
			tracker[track_id].update(centers[track_id], displacement, yaw_change)
		else:
			tracker[track_id] = Object(centers[track_id], 20)
	for track_id in tracker:# for whole ids tracked by prev frame,but current frame did not
		if track_id not in centers: # dont know its center pos
			tracker[track_id].update(None, displacement, yaw_change)
```

MOT：tracking by detection 


##### 计算两车之间的距离

min_distance_cuboids

```python

def distance_point_to_segment(P,A,B):
  """
  calculates the min distance of point P to a segment AB.
  return min distance and point q
  """

  AP = P-A
  BP = P-B
  AB = B-A
  # 锐角，投影点在线段上
  if np.dot(AB,AP)>=0 and np.dot(-AB,BP)>=0:
    return np.abs(np.cross(AP,AB))/np.linalg.norm(AB), np.dot(AP,AB)/np.dot(AB,AB)*AB+A
  # 否则线段外
  d_PA = np.linalg.norm(AP)
  d_PB = np.linalg.norm(BP)
  if d_PA < d_PB:
    return d_PA, A 
  return d_PB, B

```

##### 计算两个3D框的距离

```python

# 计算两个3d框的最短距离

def min_distance_cuboids(cub1,cub2):
  """
  compute min dist between two non-overlapping cuboids of shape (8,4)
  """

  minD = 1e5
  for i in range(4):
    for j in range(4):
      d, Q = distance_point_to_segment(cub1[i,:2], cub2[j,:2], cub2[j+1,:2])
      if d < minD:
        minD = d
        minP = ego_car[i,:2]
        minQ = Q
  for i in range(4):
    for j in range(4):
      d, Q = distance_point_to_segment(cub1[i,:2], cub2[j,:2], cub2[j+1,:2])
      if d < minD:
        minD = d
        minP = corners_3d_velo[i,:2]
        minQ = Q
  return minP, minQ, minD

```

