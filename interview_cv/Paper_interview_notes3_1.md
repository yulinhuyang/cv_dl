https://github.com/DataXujing/YOLO-v5/blob/master/yolov5_trt.py

注释版代码


前处理代码：

```python
    def preprocess_image_0(self, input_image_path):

        image_raw = cv2.imread(input_image_path)   # 1.opencv读入图片
        h, w, c = image_raw.shape                  # 2.记录图片大小
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)  # 3. BGR2RGB
        # Calculate widht and height and paddings
        r_w = INPUT_W / w  # INPUT_W=INPUT_H=640  # 4.计算宽高缩放的倍数 r_w,r_h
        r_h = INPUT_H / h
        if r_h > r_w:       # 5.如果原图的高小于宽(长边），则长边缩放到640，短边按长边缩放比例缩放
            tw = INPUT_W
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((INPUT_H - th) / 2)  # ty1=（640-短边缩放的长度）/2 ，这部分是YOLOv5为加速推断而做的一个图像缩放算法
            ty2 = INPUT_H - th - ty1       # ty2=640-短边缩放的长度-ty1
        else:
            tw = int(r_h * w)
            th = INPUT_H
            tx1 = int((INPUT_W - tw) / 2)
            tx2 = INPUT_W - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th),interpolation=cv2.INTER_LINEAR)  # 6.图像resize,按照cv2.INTER_LINEAR方法
        # Pad the short side with (128,128,128)   
        image = cv2.copyMakeBorder(
            # image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (114, 114, 114)

        )  # image:图像， ty1, ty2.tx1,tx2: 相应方向上的边框宽度，添加的边界框像素值为常数，value填充的常数值
        image = image.astype(np.float32)   # 7.unit8-->float
        # Normalize to [0,1]
        image /= 255.0    # 8. 逐像素点除255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])   # 9. HWC2CHW
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)    # 10.CWH2NCHW
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)  # 11.ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
```



注释版yolov4: https://github.com/bubbliiiing/yolov4-tiny-pytorch


Yolov4  tiny 模型： 23.1MB,  40.2% AP50, 371 FPS   <----------------->   可以对比：yolo  V5s: 27MB，400 FPS

分辨率： 416 x  416  608 x 608 

Yolo.py:  detect_image :   letterbox_image  --> net --> yolo_decodes ---> non_max_suppression---> yolo_correct_boxes

	
	
	
	

