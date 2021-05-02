# Pytorch
 let's look what things dose pytorch have.

## 优化Pytorch模型训练的小技巧

## 1.混合精度

	在一个常规的训练循环中，PyTorch以32位精度存储所有浮点数变量。模型中以16位精度存储所有变量/数字可以显著减少模型的内存消耗，加速训练循环，同时仍然保持模型的性能/精度。

在Pytorch中将所有计算转换为16位精度非常简单，只需要几行代码。这里是:

```python
 scaler = torch.cuda.amp.GradScaler()
```

	上面的方法创建一个梯度缩放标量，以最大程度避免使用fp16进行运算时的梯度下溢。

```python
optimizer.zero_grad()
 with torch.cuda.amp.autocast():
    output = model(input).to(device)
    loss = criterion(output, correct_answer).to(device)
 scaler.scale(loss).backward()
 scaler.step(optimizer)
 scaler.update()
```

	当使用loss和优化器进行反向传播时，您需要使用scale .scale(loss)，而不是使用loss.backward()和optimizer.step()。使用scaler.step(optimizer)来更新优化器。这允许你的标量转换所有的梯度，并在16位精度做所有的计算，最后用scaler.update()来更新缩放标量以使其适应训练的梯度。
	
	当以16位精度做所有事情时，可能会有一些数值不稳定，导致您可能使用的一些函数不能正常工作。只有某些操作在16位精度下才能正常工作。具体可参考官方的文档。

## 2.进度条

	有一个进度条来表示每个阶段的训练完成的百分比是非常有用的。为了获得进度条，我们将使用tqdm库。以下是如何下载并导入它:

```python
pip install tqdm
from tqdm import tqdm
```

	在你的训练和验证循环中，你必须这样做:

```python
for index, batch in tqdm(enumerate(loader), total = len(loader), position = 0, leave = True):
```

	训练和验证循环添加tqdm代码后将得到一个进度条，它表示您的模型完成的训练的百分比。它应该是这样的:![图片](https://mmbiz.qpic.cn/mmbiz_png/6wQyVOrkRNIjvaicdRJI14zgm53ibfplbtHZJEdeKJQLOvKUJ5ibopRLy2gqdiajdLgriaHxDS8zOb8DbeLwibKGan8w/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
	
	在图中，691代表我的模型需要完成多少批，7:28代表我的模型在691批上的总时间，1.54 it/s代表我的模型在每批上花费的平均时间。

## 3.梯度积累

	假设你的机器/模型只能支持16的批处理大小，增加它会导致CUDA内存不足错误，并且您希望批处理大小为32。梯度累加的工作原理是:
	
	以16个批的规模运行模型两次，将计算出的每个批的梯度累加起来，最后在这两次前向传播和梯度累加之后执行一个优化步骤。
	
	要理解梯度积累，重要的是要理解在训练神经网络时所做的具体功能：

- loss.backward()：为模型创建并存储梯度
- optimizer.step()：更新权重

在如果在调用优化器之前两次调用loss.backward()就会对梯度进行累加。下面是如何在PyTorch中实现梯度累加:

```python
model = model.train()
optimizer.zero_grad()
for index, batch in enumerate(train_loader):
    input = batch[0].to(device)
    correct_answer = batch[1].to(device)
    output = model(input).to(device)
    loss = criterion(output, correct_answer).to(device)
    loss.backward()
    if (index+1) % 2 == 0:
        optimizer.step()
        optimizer.zero_grad()
```

译者注：梯度累加只是一个折中方案，经过我们的测试，如果对梯度进行累加，那么最后一次loss.backward()的梯度会比前几次反向传播的权重高，具体为什么我们也不清楚，哈。虽然有这样的问题，但是使用这种方式进行训练还是有效果的。

16位精度的梯度累加非常类似。

```python
model = model.train()
optimizer.zero_grad()
for index, batch in enumerate(train_loader):
    input = batch[0].to(device)
    correct_answer = batch[1].to(device)
    with torch.cuda.amp.autocast():
        output = model(input).to(device)
        loss = criterion(output, correct_answer).to(device)
    scaler.scale(loss).backward()
    if (index+1) % 2 == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

## 4.结果评估

尽管计算准确率、精度、召回率和F1等指标并不困难，但在某些情况下，您可能希望拥有这些指标的某些变体，如加权精度、召回率和F1。可以使用sklearns classification_report库。这是一个专门为计算这些指标而设计的库。

```python
from sklearn.metrics import classification_report
y_pred = [0, 1, 0, 0, 1]
y_correct = [1, 1, 0, 1, 1]
print(classification_report(y_correct, y_pred))
```

# 5.统计数据集的均值和标准差

	我们在使用模型训练之前一般要对数据进行归一化(Normalize)，归一化之前需要得到数据集整体的方差和均值,其思想主要是随机从数据集采样，直接调用numpy的方法返回数据集样本的均值和方差。

```python
def get_mean_std(dataset, ratio=0.01):
    """Get mean and std by sample ratio
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset)*ratio), 
                                             shuffle=True, num_workers=10)
    train = iter(dataloader).next()[0]   # 一个batch的数据
    mean = np.mean(train.numpy(), axis=(0,2,3))
    std = np.std(train.numpy(), axis=(0,2,3))
    return mean, std
```

以下以CIFAR10数据集为例:

```python
# cifar10
train_dataset = torchvision.datasets.CIFAR10('./data', 
                                             train=True, download=False, 
                                             transform=transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10('./data', 
                                           train=False, download=False, 
                                            transform=transforms.ToTensor())
train_mean, train_std = get_mean_std(train_dataset)
test_mean, test_std = get_mean_std(test_dataset)

print(train_mean, train_std)
print(test_mean,test_std)
```