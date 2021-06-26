
### pytorch使用

#### 维度操作： 维 - 和 - 索

张量结构：维度变换(permute、squeeze、view等)、合并分割(split、chunk等)、索引切片(index_select、gather)

张量数学运算：标量、向量、矩阵

##### 维度变换

**squeeze vs unsqueeze**
	
	torch.squeeze()或者a.squeeze()
	
	squeeze:压缩维度，去掉维数是1的维度。squeeze(a)删除所有维度是1的，a.squeeze(1) 去掉a中指定维数是1。
	
	unsqueeze:数据维度扩充，指定位置加上维数是1的维度。torch.unsqueeze(a,N) 指定位置N加上一个维数是1的维度。

```python
	x = torch.rand(3,3)

	y1 = torch.unsqueeze(x,0)
	
	y2 = x.unsqueeze(0)
	
```

**transpose vs permute 维度交换**

```python
	torch.transpose 只能交换两个维度，permute可以交换任意位置
	
	input = torch.rand(1,3,28,32)
	
	print(input.permute(0,2,3,1).shape)
	
	print(input.transpose(1,3).transpose(1,2).shape)
```

##### 索引切片

**索引切片**
	
规则索引：直接切片
	
	左开右闭 规范

```python

	t = torch.randint(1,10,(3,3))
	print(t[1:,::2]
	print(1:,-1)
```
	
		
不规则索引： gather、index_select、masked_select
	
**gather** 

	不规则切片提取算子，根据指定的index
	
	[图解PyTorch中的torch.gather函数](https://zhuanlan.zhihu.com/p/352877584)
	
	[torch.gather() 和torch.sactter_()的用法简析](https://blog.csdn.net/Teeyohuang/article/details/82186666)
	
	torch.gather(input,dim,index,...)
	
	index的维度和out的维度一致；是对于out指定位置上的值，去寻找input里面对应的索引位置，根据是index
	
	dim = 0 ,输入行向量index，并替换行索引;dim = 1，输入列向量index，并替换列索引。
	
	out[i][j][k] = input[index[i][j][k]] [j][k]  # if dim == 0

	out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1

	out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

	
```python

tensor_0 = torch.arange(3, 12).view(3, 3)
print(tensor_0)

tensor([[ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])

index = torch.tensor([[2, 1, 0]])
tensor_1 = tensor_0.gather(0, index)
print(tensor_1)
--> tensor([[9, 7, 5]])
	

index = torch.tensor([[2, 1, 0]])
tensor_1 = tensor_0.gather(1, index)

print(tensor_1)
--> tensor([[5, 4, 3]])
	
	
```
	
	
**scatter_**	
	
	torch.scatter_(dim, index, src)
	
	将src中数据根据index中的索引按照dim的方向填进input中
	
	self[ index[i][j][k] ][ j ][ k ] = src[i][j][k]  # if dim == 0

	self[ i ][ index[i][j][k] ][ k ] = src[i][j][k]  # if dim == 1

	self[ i ][ j ][ index[i][j][k] ] = src[i][j][k]  # if dim == 2
	
	dim = 0 ,替换行的索引;dim = 1 ,替换列索引。
 
```python
   
	>>> x = torch.rand(2, 5)

	>>> torch.zeros(3, 5).scatter_(0, torch.LongTensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
	
	0.4319 = Src[0][0] ----->self[ index[0][0] ][0] ----> self[0][0]
	
	0.6500 = Src[0][1] ----->self[ index[0][1] ][1] ----> self[1][1]
	
	0.4080 = Src[0][2] ----->self[ index[0][2] ][2] ----> self[2][2]
```
	
**torch.index_select**
	
	torch.index_select(input，dim，index,...) -> Tensor
	
	返回沿着输入张量的指定维度的指定索引号进行索引的张量子集

	
```python	
	
	>>> indices = torch.tensor([0,2])
	>>> torch.index_select(x,0,indices)
	tensor([[-0.3902,  0.4013,  0.2772, -0.3357],
		   [ 0.4588,  1.3726, -0.2776,  0.4897]])
```


#####  合并分割
	
**torch.cat 和 torch.stack**

	torch.cat是连接，不会增加维度；torch.stack 是堆叠，会增加一个维度。
	
	torch.cat(tensors, dim=0, *, out=None) → Tensor
	
	torch.stack(tensors, dim=0, *, out=None) → Tensor

```python

    #区分range与arange

	>>> y=torch.range(1,6)
	>>> y
	tensor([1., 2., 3., 4., 5., 6.])
	>>> y.dtype
	torch.float32

	>>> z=torch.arange(1,6)
	>>> z
	tensor([1, 2, 3, 4, 5])
	>>> z.dtype
	torch.int64

	#cat 与stack
	
	>>> a = torch.arange(0,9).view(3,3)
	>>> b = torch.arange(10,19).view(3,3)
	>>> c = torch.arange(20,29).view(3,3)
	
	>>> cat = torch.cat([a,b,c],dim =0)       ----> torch.Size([9, 3])
	>>> stack_abc = torch.stack([a,b,c], axis=0)   ----> torch.Size([3, 3, 3])
	
	# # torch中dim和axis参数名可以混用
	
	chunk_abc = torch.chunk(cat_abc, 3, dim=0) 
	chunk = torch.chunk(cat,3,dim=0)           ---> 拆成3个tensor了
	
	>>> print(chunk)
	   (tensor([[0, 1, 2],
		[3, 4, 5],
		[6, 7, 8]]),
		tensor([[10, 11, 12],
		[13, 14, 15],
		[16, 17, 18]]), 
		tensor([[20, 21, 22],
		[23, 24, 25],
		[26, 27, 28]]))
```
	

**torch.split 和 torch.chunk**
	
	split和chunk都是cat的逆运算
	
	split() 作用是将张量拆分为多个块，每个块都是原始张量的视图。
	
	chunk() 作用是将tensor按dim(行或列)分割成chunks个tensor块，返回的是一个元组。
	
	torch.split(tensor, split_size_or_sections, dim=0)
	
	torch.chunk(input, chunks, dim=0) → List of Tensors
	
	>>> a = torch.arange(10).reshape(5,2)
	>>> a
	tensor([[0, 1],
	[2, 3],
	[4, 5],
	[6, 7],
	[8, 9]])
	>>> torch.split(a, 2)
	(tensor([[0, 1],
	[2, 3]]),
	tensor([[4, 5],
	[6, 7]]),
	tensor([[8, 9]]))
	>>> torch.split(a, [1,4])
    >>> torch.chunk(a, 2, dim=1)       --->切分成两块



**上采样**
	
	基于线性插值的上采样：计算效果：最近邻插值算法 < 双线性插值 < 双三次插值。计算速度：最近邻插值算法 > 双线性插值 > 双三次插值。
	
	torch.nn.functional.interpolate((input, size=None, scale_factor=None, mode='nearest'):————> int/tuple 上采样**
	
	nn.ConvTranspose2d 反卷积：转置卷积

### tensor结构：
	
	torch.Tensor 是一种包含单一数据类型元素的多维矩阵，Tensor可以使用 torch.tensor() 转换 Python 的 list 或序列数据生成。
	
	a.指定数据类型的 Tensor 可以通过传递参数 torch.dtype 和/或者 torch.device 到构造函数生成
	
```python
	>>> torch.ones([2,3], dtype=torch.float64, device="cuda:0")
	tensor([[1., 1., 1.],1., 1., 1.]], device='cuda:0', dtype=torch.float64)
```

	b.Tensor内容可以通过python索引或者切片访问以及修改
	
```python	
	>>> martix = torch.tensor([[2,3,4],[5,6,7]])
	print(martix[1][2])
```	
	c.使用 torch.Tensor.item() 或者 int() 方法从只有一个值的 Tensor中获取 Python Number：
	
```python 

	>>> x = torch.tensor([[4.5]])
	>>> x
	tensor([[4.5000]])
	>>> x.item()
	4.5
	>>> int(x)
	4
```
	d.Tensor可以通过参数 requires_grad=True 创建, 这样 torch.autograd 会记录相关的运算实现自动求导

	torch.Tensor 是默认的 tensor 类型（ torch.FloatTensor ）的简称，即 32 位浮点数数据类型
  
  
	
#### tensor的属性

	维度：可以使用dim()方法获取tensor的维度
	
	尺寸：shape属性 或者 size()方法 查看张量在每一维的长度，可以使用view或reshape改变张量的尺寸
	
	view()只对满足连续性条件的tensor进行操作，reshape()同时可以对不满足连续性条件的tensor进行操作。
	
	tensor满足连续性条件时(contiguous),a.reshape的结果与a.view相同;不满足contiguous时，只能使用reshape(),重开内存，返回副本。
	
	
#### Tensor 与 ndarray

	tensor -> numpy: .numpy()
	
	numpy -> tensor: torch.from_numpy()
	
	上面两种都是共享内存的，可以使用张量的.clone拷贝张量，中断关联。
	
```python 

	arr = np.random.rand(4,5)
	print(type(arr))
	tensor1 = torch.from_numpy(arr)
	print(type(tensor1))
	arr1 = tensor1.numpy()
	print(type(arr1))
	"""
	<class 'numpy.ndarray'>
	<class 'torch.Tensor'>
	<class 'numpy.ndarray'>
	"""
```	
	item() 方法和 tolist() 方法可以将张量转换成 Python 数值和数值列表

```python
	
	scalar = torch.tensor(5) # 标量
	s = scalar.item()
	t = tensor.tolist()
	print(t)
	print(type(t))

```

#### 创建tensor

	torch.rand(*sizes,out=None)          ---> 区间[0,1)均匀分布中抽样
		
	torch.randint(low=0, high,size)      ---> 指定范围(low,high)和size的随机数据
	
	torch.arange(start=0, end, step=1, *, out=None)  --->生成指定间隔的数据
  
  
