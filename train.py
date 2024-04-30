from model import *
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default= './dataset')
    parser.add_argument('--logpath', default='./results/test')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=4396)
    parser.add_argument('--epoch', type=int, default=100)
    #parameters
    parser.add_argument('--hidden_dim', type=int, default=300, help = '网络隐藏层维度')
    parser.add_argument('--L2_lambda', type=float, default=0.01, help = 'L2正则化强度')
    parser.add_argument('--lr', type=float, default=0.5, help = '初始学习率')
    parser.add_argument('--decay_epoch', type=int, default=80, help = '学习率衰减周期')
    parser.add_argument('--lr_decay', type=float, default=0.5, help = '学习率衰减率')

    args = parser.parse_args()
    if os.path.exists(args.logpath) == 0:
        os.makedirs(args.logpath)
    set_log_path(args.logpath)
    name = 'hd{}_reg{}_lr{}'.format(args.hidden_dim, args.L2_lambda, args.lr)
    filename = '{}.txt'.format(name)
    log(args, filename)
    images, labels = load_mnist(args.datapath, kind='train')
    test_images, test_labels = load_mnist(args.datapath, kind='t10k')
    layer_dims = [images.shape[0], args.hidden_dim, 10]
    results = train(images, labels, test_images, test_labels, layer_dims, learning_rate=args.lr,
                    lr_decay=args.lr_decay, decay_epoch=args.decay_epoch, L2_lambda=args.L2_lambda,
                    mini_batch_size=args.batch_size, num_epochs=args.epoch, name=name)
    save_filename = 'checkpoint_{}.pkl'.format(name)
    pickle_dump(results, save_filename)