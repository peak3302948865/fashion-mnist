from model import *
import matplotlib.pyplot as plt
import copy

datapath = './dataset'
logpath = './results'
seed = 4396
batch_size = 128
num_epochs = 100
decay_epoch = 80
lr_decay = 0.5
L2_lambdas = [0, 1e-1, 1e-2, 1e-3, 1e-4]
hidden_dims = [800, 500,300,100,50]
lrs = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]


if __name__ == '__main__':
    if os.path.exists(logpath) == 0:
        os.makedirs(logpath)
    set_log_path(logpath)
    images, labels = load_mnist(datapath, kind='train')
    test_images, test_labels = load_mnist(datapath, kind='t10k')
    filename = 'select_parameters.log'

    log('开始参数选择', filename)
    log('#' * 30, filename)
    log('L2_lambdas:', filename)
    log(L2_lambdas, filename)
    log('hidden_dims:', filename)
    log(hidden_dims, filename)
    log('learning rates:', filename)
    log(lrs, filename)
    log('#'*30, filename)


    best_model = {}
    best_name = ''
    best_accuracy = 0

    for hidden_dim in hidden_dims:
        for L2_lambda in L2_lambdas:
            for lr in lrs:
                name = 'hd{}_reg{}_lr{}'.format(hidden_dim, L2_lambda, lr)
                log('#' * 20, filename)
                log('current parameters:{}'.format(name), filename)
                layer_dims = [images.shape[0], hidden_dim, 10]
                results = train(images, labels, test_images, test_labels, layer_dims, learning_rate=lr,
                                lr_decay=lr_decay, decay_epoch = decay_epoch, L2_lambda=L2_lambda,
                                mini_batch_size=batch_size, num_epochs=num_epochs, name = name)
                accuracy = np.max(results['accuracy_test'])
                log('当前测试集分类正确率:{:.2%}'.format(accuracy), filename)
                save_filename = 'checkpoint_{}.pkl'.format(name)
                pickle_dump(results, save_filename)
                if accuracy > best_accuracy:
                    log('更新最佳模型', filename)
                    best_accuracy = accuracy
                    best_model = results
                    best_name = name
                log('#' * 20, filename)

    save_filename = 'checkpoint_best_{}.pkl'.format(best_name)
    pickle_dump(best_model, save_filename)
    log('保存最佳模型', filename)
    log('结束参数选择', filename)
