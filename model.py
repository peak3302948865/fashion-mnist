# -*- coding: utf-8 -*-

import os
import struct
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
import copy

_log_path = None

################保存&读取模型#####################
def set_log_path(path):
    global _log_path
    _log_path = path

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)

def pickle_dump(object, filename='save.pkl'):
    with open(os.path.join(_log_path, filename), 'wb') as file:
        pickle.dump(object, file)

def pickle_read(filename='save.pkl'):
    with open(os.path.join(_log_path, filename), 'rb') as file:
        data = pickle.load(file)
        return data

#################读取&预处理数据#################
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def normalize(X):
    X = X.astype('float64')
    return X/np.max(X)

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,'{}-labels.idx1-ubyte'.format(kind))
    images_path = os.path.join(path,'{}-images.idx3-ubyte'.format(kind))
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    labels = convert_to_one_hot(labels, 10)
    return images.T, labels

def random_mini_batches(X, Y, mini_batch_size=64, seed=1234):
    """
    从（X，Y）中创建一个随机的mini-batch列表

    参数：
        X - 输入数据，维度为(输入节点数量，样本的数量)
        Y - 对应的是X的标签
        mini_batch_size - 每个mini-batch的样本数量

    返回：
        mini-bacthes - 一个同步列表，维度为（mini_batch_X,mini_batch_Y）

    """

    np.random.seed(seed)  # 指定随机种子
    m = X.shape[1]
    mini_batches = []

    # 第一步：打乱顺序
    permutation = list(np.random.permutation(m))  # 它会返回一个长度为m的随机数组，且里面的数是0到m-1
    shuffled_X = X[:, permutation]  # 将每一列的数据按permutation的顺序来重新排列。
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # 第二步，分割
    num_complete_minibatches = math.floor(m / mini_batch_size)  # 把你的训练集分割成多少份,请注意，如果值是99.99，那么返回值是99，剩下的0.99会被舍弃
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # 如果训练集的大小不是mini_batch_size的整数倍，那么最后肯定会剩下一些，要把它处理了
    if m % mini_batch_size != 0:
        # 获取最后剩余的部分
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

################激活函数#####################
def sigmoid(x):
    """
    Compute the sigmoid of x
 
    Arguments:
    x -- A scalar or numpy array of any size.
 
    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s
 
def relu(x):
    """
    Compute the relu of x
 
    Arguments:
    x -- A scalar or numpy array of any size.
 
    Return:
    s -- relu(x)
    """
    s = np.maximum(0,x)
    
    return s

def softmax(x):
    """
    Implements the softmax activation in numpy

    Arguments:
    x -- numpy array of any shape, dims x batch_size

    Returns:
    s -- output of sigmoid(z), same shape as x
    """
    #防止溢出
    M = np.max(x, axis=0, keepdims = True)
    #print(M)
    x = x-M
    epsilon = 1e-8
    s = np.exp(x) / (epsilon+np.sum(np.exp(x), axis=0, keepdims=True))

    return s

######################初始化网络参数####################
def initialize_parameters(layer_dims, seed = 1234):
    """
    参数:
    layer_dims -- 隐藏层节点数量列表
    
    返回:
    parameters - 一个字典，包含了以下参数：
            parameters["W" + str(l)] = Wl
            parameters["b" + str(l)] = bl

    """
    
    np.random.seed(seed)
    parameters = {}
    L = len(layer_dims) # number of layers in the network
 
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.normal(0.0, np.sqrt(2 / layer_dims[l-1]), (layer_dims[l], layer_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters

######################2层神经网络的前向传播####################
def forward_propagation(X, parameters):
    """
    输入：
    X -- 输入数据，维度为(输入节点数量，样本的数量)
    parameters -- 保存参数的字典
    
    返回:
    前向传播的结果A2和保存中间值cache
    """
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = softmax(z2)

    cache = (z1, a1, W1, b1, z2, a2, W2, b2)
    
    return a2, cache


######################2层神经网络的反向传播####################
def backward_propagation(X, Y, cache, L2_lambda):
    """
    2层网络的反向传播，带有L2正则化
    
    输入：
    X -- 输入数据，维度为(输入节点数量，样本的数量)
    Y -- X对应的标签（one-hot向量）
    cache -- 前向传播中传递的值
    
    返回:
    gradients -- 包含参数对应梯度的字典
    """
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2) = cache
    
    dz2 = 1./m * (a2 - Y)
    dW2 = np.dot(dz2, a1.T) + (L2_lambda*W2)/m
    db2 = np.sum(dz2, axis=1, keepdims = True)
    
    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T) + (L2_lambda*W1)/m
    db1 = np.sum(dz1, axis=1, keepdims = True)
    
    gradients = {"dW2": dW2, "db2": db2,
                 "dW1": dW1, "db1": db1}
    
    return gradients

######################计算Loss####################
def compute_cost(a2, Y, parameters,L2_lambda):
    
    """
    计算loss
    """
    m = Y.shape[1]
    epsilon = 1e-8
    logprobs = np.multiply(-np.log(a2+epsilon),Y)
    cost = 1./m * np.sum(logprobs)

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    cost = cost+(np.sum(np.square(W1))+np.sum(np.square(W2)))*L2_lambda/(2*m)
    
    return cost


def update_parameters_with_sgd(parameters, grads, learning_rate):
    """
    使用sgd更新参数

    参数：
        parameters - 字典，包含了要更新的参数：
            parameters['W' + str(l)] = Wl
            parameters['b' + str(l)] = bl
        grads - 字典，包含了每一个梯度值用以更新参数
            grads['dW' + str(l)] = dWl
            grads['db' + str(l)] = dbl
        learning_rate - 学习率

    返回值：
        parameters - 字典，包含了更新后的参数
    """

    L = len(parameters) // 2  # 神经网络的层数

    # 更新每个参数
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters

#####################################调整学习率##############
def adjust_learning_rate(lr, decay = 0.1):
    return lr*decay

######################测试正确率#################
def test(test_X, test_Y, parameters, L2_lambda=0):
    # 归一化
    test_X = normalize(test_X)
    A2, cache = forward_propagation(test_X, parameters)
    cost = compute_cost(A2, test_Y, parameters, L2_lambda)

    pred_Y = np.argmax(A2, axis = 0)
    test_Y = np.argmax(test_Y, axis= 0)
    accuracy = np.mean(test_Y == pred_Y)
    return accuracy, cost

######################整体的训练函数#################
def train(X, Y, test_X, test_Y, layers_dims, seed=1234,
          learning_rate=0.001, lr_decay=0.5, decay_epoch = 50, L2_lambda=0,
          mini_batch_size=64, num_epochs=200, name = '000'):
    """
    2层神经网络的训练函数

    参数：
        X - 输入数据，维度为（2，输入的数据集里面样本数量）
        Y - 与X对应的标签
        layers_dims - 包含层数和节点数量的列表
        learning_rate - 学习率
        mini_batch_size - 每个小批量数据集的大小
        num_epochs - 整个训练集的遍历次数
        print_cost - 是否打印误差值
        is_plot - 是否绘制出曲线图

    返回：
        parameters - 包含了学习后的参数

    """
    L = len(layers_dims)
    costs_train = []
    costs_test = []
    accuracy_train = []
    accuracy_test = []
    t = 0  # 每学习完一个minibatch就增加1

    # 初始化参数
    parameters = initialize_parameters(layers_dims, seed)
    best_parameters = {}
    best_accuracy = 0

    filename = '{}.txt'.format(name)  # log文件名
    log('开始训练', filename)

    t = 0  # 全局步数
    # 开始学习
    for i in range(num_epochs):
        # 定义随机 minibatches,在每次遍历数据集之后增加种子以重新排列数据集，使每次数据的顺序都不同
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed+t)

        for minibatch in minibatches:
            t = t + 1
            # 选择一个minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # 归一化
            minibatch_X = normalize(minibatch_X)

            # 前向传播
            A2, cache = forward_propagation(minibatch_X, parameters)
            # print(A3)

            # 计算误差
            cost = compute_cost(A2, minibatch_Y,parameters, L2_lambda)
            #costs_train.append(cost)

            # 反向传播
            grads = backward_propagation(minibatch_X, minibatch_Y, cache, L2_lambda)



            # 更新参数
            parameters = update_parameters_with_sgd(parameters, grads, learning_rate)


            del minibatch

        del minibatches

        accuracy, cost= test(X, Y, parameters, L2_lambda)
        accuracy_train.append(accuracy)
        costs_train.append(cost)
        log("第{}次遍历整个数据集，当前训练集正确率：{:.2%}，loss：{:.4f}".format(i + 1, accuracy, cost), filename)

        # 在测试集上进行测试
        accuracy, cost= test(test_X, test_Y, parameters, L2_lambda)
        accuracy_test.append(accuracy)
        costs_test.append(cost)
        log("测试集正确率：{:.2%}，loss：{:.4f}".format(accuracy, cost), filename)
        if accuracy > best_accuracy:
            best_parameters = parameters
            best_accuracy = accuracy
            log("更新最佳模型", filename)

        if decay_epoch >0 and (i+1) % decay_epoch == 0:
            learning_rate = adjust_learning_rate(learning_rate, decay=lr_decay)
            log('调整学习率至{}'.format(learning_rate), filename)

    #save_filename = 'checkpoint_{}.pkl'.format(name)
    object = {'parameter': best_parameters,
              'loss_train': costs_train, 'accuracy_train': accuracy_train,
              'loss_test': costs_test, 'accuracy_test': accuracy_test}
    #pickle_dump(object, save_filename)
    log('训练结束', filename)

    return object
