import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from custom_dataset import CustomDataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # 网络构造代码

    def forward(self, x):
        # 网络前向传播代码
        return x


if __name__ == '__main__':

    # 超参数
    learning_rate = 0.01
    epochs = 10
    batch_size = 64

    #数据集
    dataSet = CustomDataset()
    train_features = dataSet.GetTrainFeature()
    test_features = dataSet.GetTestFeature()
    train_labels = dataSet.GetTrainLabel()
    # 封装到TensorDataset
    train_data = TensorDataset(train_features, train_labels)
    test_data = TensorDataset(test_features)
    # 使用DataLoader进行加载
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # 分割数据集为训练集和测试集
    features_train, features_test, labels_train, labels_test = train_test_split(train_features, train_labels, test_size=0.2,
                                                                                random_state=42)

    # 创建随机森林回归模型
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)

    # 训练模型
    regressor.fit(features_train, labels_train)

    # 做预测
    predictions = regressor.predict(features_test)

    # 计算预测结果的均方误差
    mse = mean_squared_error(labels_test, predictions)
    print("Mean Squared Error: ", mse)




    # model = MyNet()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #
    # for epoch in range(epochs):
    #     for i, data in enumerate(dataloader, 0):
    #         inputs, labels = data
    #         optimizer.zero_grad()
    #
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    # # 保存模型
    # torch.save(model.state_dict(), 'model.pt')

    # 加载模型
    # model = Net()
    # model.load_state_dict(torch.load('model.pt'))