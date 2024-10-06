import torch
from torch import nn
from d2l import torch as d2l


if __name__ == '__main__':
    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    a=train_iter.dataset.__dict__.values().s
    print(str(a))