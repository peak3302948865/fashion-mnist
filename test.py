import numpy as np
import matplotlib.pyplot as plt
import argparse
from model import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default='./dataset')
    parser.add_argument('--logpath', default='./results')
    parser.add_argument('--checkpoint', default='checkpoint_best_hd300_reg0.01_lr0.5.pkl')
    args = parser.parse_args()
    set_log_path(args.logpath)
    #读取数据
    test_images, test_labels = load_mnist(args.datapath, kind='t10k')
    data = pickle_read(args.checkpoint)
    parameters = data['parameter']
    accuracy, cost = test(test_images, test_labels, parameters)
    print('该模型在测试集上的分类正确率为：{:.2%}'.format(accuracy))


    #可视化损失函数
    epoch = len(data['loss_train'])
    x = np.linspace(1, epoch, epoch)
    plt.figure(1)
    ax1 = plt.subplot(111)
    ax1.plot(x, data['loss_train'], label='train')
    ax1.plot(x, data['loss_test'], label='test')
    ax1.set_yscale('log')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_title('Loss in Train and Test Dataset')
    ax1.legend()

    #可视化分类正确率
    plt.figure(2)
    ax3 = plt.subplot(111)
    ax3.plot(x, data['accuracy_train'], label='train')
    ax3.plot(x, data['accuracy_test'], label='test')
    #ax3.set_yscale('log')
    ax3.set_xlabel('epoch')
    ax3.set_ylabel('accuracy')
    ax3.set_title('Accuracy in Train and Test Dataset')
    ax3.legend()

    #可视化网络每层的参数
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    b1 = np.repeat(b1, b1.shape[0], axis=1)
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    b2 = np.repeat(b2, b2.shape[0], axis=1)
    plt.figure(3)
    axs = plt.subplot(221)
    axs.imshow(W1)
    axs.set_title('W1')
    plt.xticks([]), plt.yticks([])
    axs = plt.subplot(222)
    axs.imshow(b1)
    axs.set_title('b1')
    plt.xticks([]), plt.yticks([])
    axs = plt.subplot(223)
    axs.imshow(W2)
    axs.set_title('W2')
    plt.xticks([]), plt.yticks([])
    axs = plt.subplot(224)
    axs.imshow(b2)
    axs.set_title('b2')
    plt.xticks([]), plt.yticks([])
    plt.show()

