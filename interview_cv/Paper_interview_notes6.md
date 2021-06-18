## yolov5 代码解读


[YOLOv5代码详解（train.py部分）](https://blog.csdn.net/mary_0830/article/details/107076617)

[YOLOv5代码详解（yolov5l.yaml部分）](https://blog.csdn.net/mary_0830/article/details/107124459)



### yaml解读

s->m->l->x      smlx

nc：类别数，你的类别有多少就填写多少。从1开始算起，不是0-14这样算。

depth_multiple：控制模型的深度。

width_multiple：控制卷积核的个数。


        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]           控制宽度
            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # number of repeats，控制深度
                n = 1

###训练代码


**1 输入端：**

Mosaic数据增强、自适应锚框计算、自适应图片缩放

	noautoanchor：但Yolov5中将此功能嵌入到代码中，每次训练时，自适应的计算不同训练集中的最佳锚框值。

	letterbox函数中进行了修改，对原始图像自适应的添加最少的黑边。

**2 Backbone：**

Focus结构：切片操作，类似passthrough

CSP结构：CSP1_X结构应用于Backbone主干网络，另一种CSP2_X结构则应用于Neck中

**3 Neck：**

FPN+PAN结构

**4 Prediction：**

GIOU_Loss，DIOU_nms


大图小目标检测：对大分辨率图片先进行分割，变成一张张小图，再进行检测。为了避免两张小图之间，一些目标正好被分割截断，所以两个小图之间设置overlap重叠区域，比如分割的小图是960*960像素大小，则overlap可以设置为960*20%=192像素。


ModelEMA：

	Model Exponential Moving Average，近期数据更高权重的平均方法

	[指数移动平均（EMA）的原理及PyTorch实现](https://www.jianshu.com/p/f99f982ad370)

	模型权重在最后的n步内，会在实际的最优点处抖动，所以我们取最后n步的平均，能使得模型更加的鲁棒

DDP训练：

		# DP mode
		if cuda and rank == -1 and torch.cuda.device_count() > 1:
			model = torch.nn.DataParallel(model)

		# SyncBatchNorm
		if opt.sync_bn and cuda and rank != -1:
			model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
			logger.info('Using SyncBatchNorm()')

		主进程，rank 等于0或者-1

混合精度训练: scaler = amp.GradScaler(enabled=cuda)
