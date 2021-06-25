[Triplet Loss 和 Center Loss详解和pytorch实现](https://blog.csdn.net/weixin_40671425/article/details/98068190)

[人脸识别损失函数简介与Pytorch实现：ArcFace、SphereFace、CosFace](https://zhuanlan.zhihu.com/p/60747096)


### center loss 原理

				   m
	公式：Lc=1/2 * ∑ || Xi- Cyi||^2
				   i=1
	
	聚类中中心的数量，等于num_classes的数量
	
	通过学习每个类的类中心，使得类内的距离变得更加紧凑。
	
	torch.randint(3, 5, (1,3)) ----> (3,5)之间的，维度是 1行，3列

	
	https://github.com/KaiyangZhou/pytorch-center-loss

	
```python

	def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
		super(CenterLoss, self).__init__()
		self.num_classes = num_classes
		self.feat_dim = feat_dim
		self.use_gpu = use_gpu

		if self.use_gpu:
			self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
		else:
			self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

	def forward(self, x, labels):
		"""
		Args:
			x: feature matrix with shape (batch_size, feat_dim).
			labels: ground truth labels with shape (batch_size).
		"""
		batch_size = x.size(0)
		distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
				  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
		distmat.addmm_(1, -2, x, self.centers.t())

		classes = torch.arange(self.num_classes).long()
		if self.use_gpu: classes = classes.cuda()
		labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
		mask = labels.eq(classes.expand(batch_size, self.num_classes))

		dist = distmat * mask.float()
		loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
	
		
更新：
	
# features (torch tensor): a 2D torch float tensor with shape (batch_size, feat_dim)
# labels (torch long tensor): 1D torch long tensor with shape (batch_size)
# alpha (float): weight for center loss
loss = center_loss(features, labels) * alpha + other_loss
optimizer_centloss.zero_grad()
loss.backward()
# multiple (1./alpha) in order to remove the effect of alpha on updating centers
for param in center_loss.parameters():
	param.grad.data *= (1./alpha)
optimizer_centloss.step()

```
	

### triplet loss原理 

	减少positive（正样本）与anchor之间的距离，扩大negative（负样本）与anchor之间的距离
	
	D(a,p) < D(a,n)  ---> D(a, p) + margin  <  D(a, n)
	
	|| f(Xa) - f(Xp)||^2 + a < || f(Xa) - f(Xn)||^2
	
	positive pair <a, p>和negative pair <a, n>
	
	a: anchor 表示训练样本。

	p: positive 表示预测为正样本。

	n: negative 表示预测为负样本。
	
	triplet学习的是样本间的相对距离，没有学习绝对距离，尽管考虑了类间的离散性，但没有考虑类内的紧凑性
	
	训练时：早期为了网络loss平稳，一般选择easy triplets进行优化，后期为了优化训练关键是要选择hard triplets
	
	源码解读： distance(a,b) = a*a + b*b - 2*a*b

	torch.sum()： 
	
		torch.sum(input, list: dim, bool: keepdim=False, dtype=None) → Tensor，求和之后这个dim的元素个数为１，所以要被去掉，如果要保留这个维度，则应当keepdim=True
	
		dim=0，降维（纵向压缩）。dim=1，降维（横向压缩）。
	
	addmm_：两个矩阵乘， torch.addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None) ，out = beta*mat1 + alpha*mat2
	
	torch.manual_seed(1024)  设置随机种子
	
	tensor.clamp(min=1e-12)： min(max(x,min_value),max_value)


```python

	def __init__(self, margin=0.3,global_feat, labels):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
		
		n = inputs.size(0)

		# Compute pairwise distance, replace by the official when merged
		dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
		dist = dist + dist.t() # a*a + b*b

		dist.addmm_(1, -2, inputs, inputs.t()) # a*a + b*b - 2*a*b
		dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

		# For each anchor, find the hardest positive and negative
		mask = targets.expand(n, n).eq(targets.expand(n, n).t())
		print(mask)
		dist_ap, dist_an = [], []
		for i in range(n):
			dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))        #ap的最大，此处就是挖掘
			dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))   #an的最小，此处就是挖掘
		dist_ap = torch.cat(dist_ap)
		dist_an = torch.cat(dist_an)

		# Compute ranking hinge loss
		y = torch.ones_like(dist_an)
		return self.ranking_loss(dist_an, dist_ap, y)
```

```python

	hard_example_mining相关
	
	https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/modeling/losses/triplet_loss.py
	
	def euclidean_dist(x, y):
		m, n = x.size(0), y.size(0)
		xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
		yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
		dist = xx + yy - 2 * torch.matmul(x, y.t())
		dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
		return dist


	def cosine_dist(x, y):
		x = F.normalize(x, dim=1)
		y = F.normalize(y, dim=1)
		dist = 2 - 2 * torch.mm(x, y.t())
		return dist
		
	def hard_example_mining(dist_mat, is_pos, is_neg):
	
		# `dist_ap` means distance(anchor, positive)
		# both `dist_ap` and `relative_p_inds` with shape [N]
		dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
		# `dist_an` means distance(anchor, negative)
		# both `dist_an` and `relative_n_inds` with shape [N]
		dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)
		
		y = dist_an.new().resize_as_(dist_an).fill_(1)
```

### Softmax 
	
	softmax归一化指数函数, 将一组向量进行压缩，使得到的向量各元素之和为 1,在实际运算的时候，为了避免上溢和下溢，在将向量丢进softmax之前往往先对每个元素减去其中最大值
	
	softmax -->   softmax loss -->cross Entropy
	
							K
	exp(xi)/∑j exp(xj) ---> ∑ -yk *log(softmax)
							k=1
							
	softmax loss = crossEntropy(softMax)
	
	yk 长度为K的one-hot向量
