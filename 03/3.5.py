import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from torch.utils.data import DataLoader

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间


if __name__== '__main__':
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    print(f"mnist_train {str(mnist_train)}\nmnist_test {str(mnist_test)}\n")

    # 创建一个 DataLoader 实例
    dataloader = DataLoader(mnist_train, batch_size=10, shuffle=True)

    # 获取一个批次的数据
    images, labels = next(iter(dataloader))

    # 打印图像和标签的形状
    print('Images shape:', images.shape)
    print('Labels shape:', labels.shape)