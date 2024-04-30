# fashion-mnist
## 作业：构建两层神经网络分类器
课程**DATA620004 神经网络和深度学习**作业1相关代码

周杭琪 21110980019


## Dependencies
numpy>=1.16.4


## Code organization
代码框架主要如下：

* `model.py` 主要包含神经网络的模型主体，包括数据读取、激活函数、前向传播、反向传播（梯度计算，添加L2正则化），loss计算、梯度更新（SGD）、学习率下降、训练函数和测试函数
* `parameter_selection.py` 选择网络参数（学习率，隐藏层大小，正则化强度）并筛选出效果最好的模型
* `train.py` 自定义训练并保存模型
* `test.py` 测试保存的模型并进行相关数据的可视化


## Run experiments
### 准备数据与预训练模型
* 下载代码至本地

* 下载[MNIST数据集](https://pan.baidu.com/s/16Odcj03UzZ7hGnr4ztIGdg)（提取码：q6yg）至本地，将其解压并移动到`./dataset`文件夹中

* 下载[训练好的模型](https://pan.baidu.com/s/1vKV-wPI-vj5uxo-Sh0GRfg)（提取码：gxtp）至本地，将其解压移动到`./results`文件夹中

### 快速测试
运行下列代码可以迅速测试预训练模型在测试集上的分类精度，并进行损失函数、分类精度与网络参数的可视化，获得报告中的结果
```
python  test.py
python  test.py  --datapath ./dataset --logpath ./results --checkpoint checkpoint_best_hd300_reg0.01_lr0.5.pkl
```
* --datapath：存放测试数据的文件夹路径
* --logpath：存放预训练模型的文件夹路径
* --checkpoint：预训练模型的文件名（.pkl）

### 参数选择
在进行参数选择前，可以修改parameter_selection.py中对应的参数范围，如：
```
L2_lambdas = [0, 1e-1, 1e-2, 1e-3, 1e-4]
hidden_dims = [800, 500,300,100,50]
lrs = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
```
之后运行如下代码即可进行参数选择：
```
python  parameter_selection.py
```
训练的log与模型默认保存在`./results`文件夹中，也可通过修改文件中对应的`logpath`变量来自定义保存路径

### 自定义训练
也可以运行`train.py`来训练并保存自定义参数的单个模型：
```
python  train.py
python  train.py --datapath ./dataset --logpath ./results --batch_size 128 --seed 4396 --epoch 100 --hidden_dim 300 --L2_lambda 0.01 --lr 0.5 --decay_epoch 80 --lr_decay 0.5
```
* --datapath：存放训练数据的文件夹路径
* --logpath：保存模型的文件夹路径
* --batch_size：批大小
* --seed：训练的随机数种子
* --epoch：训练的总epoch数
* --hidden_dim：网络隐藏层维度
* --L2_lambda：L2正则化强度
* --lr：初始学习率
* --decay_epoch：学习率衰减周期
* --lr_decay：学习率衰减率
