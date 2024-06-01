import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
import torch
from torch import nn, optim




def load_array(data_arrays,batch_size,is_train=True):
    data_set =  data.TensorDataset(*data_arrays)
    return data.DataLoader(data_set,batch_size,shuffle=is_train)


if __name__== '__main__':
    #0.超参数准备
    lr=0.1
    num_epochs=3

    # 1.数据准备
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1005)
    biz_size = 50
    data_loader = load_array((features, labels), biz_size)

    #2.模型选择，线性模型
    model = nn.Sequential(nn.Linear(2,1))

    #3.损失函数
    loss_func=nn.MSELoss()

    #4.优化器，根据loss优化参数
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)

    #5.模型训练
    for epoch in range(num_epochs):
        for data,target in data_loader:
            output=model(data)
            epoch_loss=loss_func(output,target)
            optimizer.zero_grad()
            epoch_loss.backward()
            optimizer.step()
        epoch_loss=loss_func(model(features),labels)
        print(f"epoch:{epoch}  epoch_loss:{epoch_loss}")
    weights = model[0].weight.data
    b=model[0].bias.data
    print(f"weights:{weights},b:{b}")

