
Yolov4  tiny 模型： 23.1MB,  40.2% AP50, 371 FPS   <----------------->   可以对比：yolo  V5s: 27MB，400 FPS

分辨率： 416 x  416  608 x 608 

Yolo.py:
detect_image :   letterbox_image  --> net --> yolo_decodes ---> non_max_suppression---> yolo_correct_boxes

模型参考：

https://github.com/bubbliiiing/yolov4-tiny-pytorch


#### 数据处理：

鱼眼镜头，标注格式， 

image_name 225,226,292,444,1,0,0,0  236,57,297,282,1,0,0,1 236,226,292,282,-1,-0.9979546466137569,-0.06392592042365702,0 

image_name + box(left, top, right, bottom) + 类别（0,1,2分别为入口线框的水平、垂直、倾斜；-1为角点）+ 角度（cos,sin;入口线框这两位为0）+  占用（1为占用；角点该位为0）

处理流程：

标注数据拷贝  ---> 原始图片上下左右外扩28 ----> json 读取，处理后转存txt,（根据三个角点，计算角度，判断垂直、水平和倾斜车位，判断是否是在驶入边）--> 制作训练用的txt

角度信息：只有角点。角点中间线，和图像X轴夹角，范围是-180~180。

特殊数据：夜景、雨天、嵌草砖场景 ---> 加数据

数据增强：常规增强翻转、放缩、色域变换等。


#### 模型优化

预测slot的head: P4、P5层，输出9个数： x y w h  conf (是否有物体) num_classes（3） occupy
	
预测的mark的head:P5_local层（3层conv）, 输出7个数：x  y  w  h  conf (是否有物体) + angleX 、angleY

两组anchor：240,240,40,240,240,40 slot  +   360,360,60,360,360,60 slot +  56, 56,28,28,84, 84 mark

所以输出大小为：P4(3*9=27，1 27 26 26)，P5(1 27 13 13),P5_local(1 3*7 13 13)

后处理流程：
	
	四象限假设：根据一个点一个框，计算另外一个点。框必然包住了两个点，如果一个框在第一象限，另外一个必然在第四象限(对角象限)

	遍历点和框 -> 根据一个mark点计算另外一个，根据两个mark点之间距离判断长短边--> 根据mark 1和 2的向量判断属于顺时针或者逆时针---> 判断入口线，计算点3,4---> 判断是否超过图像阈值--->存入list
	
	相机标定，cam2ego坐标系换算
	
	--->   
	|
	|
	
	point_ego.y = (0.5* image_width - x_image)/egoRatio
	
	point_ego.x = (0.5* image_height - y_image)/egoRatio + egoBias
	
	800*800：1个像素对应2cm，1m = 50 * 2cm

	416*416: 1个像素对应4cm, 1m = 25 * 4cm
	


cos sin 原因：方便直接回归和收敛，如果直接回归绝对角度会有0-360度跳转的问题

loss计算： 三个head分支的 YOLOLoss_local + YOLOLoss + YOLOLoss

YOLOLoss： conf loss（BCEloss mask + noobj_mask） + cls loss(BCEloss mask + noobj_mask) + loc loss(CIOU) +  loss_occupy(MSE loss（求和))

YOLOLoss_local： conf loss（BCEloss mask + noobj_mask） + angle loss(MSE loss) + loc loss(CIOU) 

尝试无效的：mosaic、attention(se CBAM)


#### 部署指标

	map： entrance_ap：0.713 ，marker_ap: 0.816

	map计算的时候：只会计算角度偏差在一定范围内的，认为是正确的。
	




