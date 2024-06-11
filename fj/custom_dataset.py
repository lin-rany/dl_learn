import torch
from torch.utils.data import Dataset
import pandas as pd

is_test = True

class CustomDataset(Dataset):
    def __init__(self,train_csv_file='../data/house-prices-advanced-regression-techniques/train.csv',
                 test_csv_file='../data/house-prices-advanced-regression-techniques/test.csv'):
        # 读取csv文件
        train_dataset = pd.read_csv(train_csv_file)
        test_dataset = pd.read_csv(test_csv_file)
        # 获取对应的feature list
        train_feature = train_dataset.iloc[:, 1:-1]
        test_feature = test_dataset.iloc[:, 1:]
        # 链接在一块，不然get_dummies后，测试集和训练集的特征大小将不一样
        all_feature = pd.concat((train_feature,test_feature))

        if is_test:
            print(f"all_feature shape: {all_feature.shape}")

        # 假设前n列特征，最后一列是标签
        self.all_features = all_feature
        # 若无法获得测试数据，则可根据训练数据计算均值和标准差
        numeric_features = self.all_features.dtypes[self.all_features.dtypes != 'object'].index
        self.all_features[numeric_features] = self.all_features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std()))
        # 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
        self.all_features[numeric_features] = self.all_features[numeric_features].fillna(0)
        # “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
        self.all_features = pd.get_dummies(self.all_features, dummy_na=True)

        # Convert all_features to float32 type
        self.all_features = self.all_features.astype('float32')
        if is_test:
            print(f"all_features shape: {self.all_features.shape}")

        self.train_features = self.all_features.iloc[:len(train_dataset)]
        self.test_features = self.all_features.iloc[len(train_dataset):]

        self.train_labels = train_dataset.iloc[:,-1].values.reshape(-1, 1)
        # if is_test:
        #     print(f"train_labels shape: {self.train_labels.shape}")
        #     print(f"train_labels[0:5]: {self.train_labels[0:5]}")

        # self.train_labels = self.train_labels.astype('float32')


    def __len__(self):
        return len(self.all_features)

    def __getitem__(self, index):
        return torch.tensor(self.all_features.iloc[index], dtype=torch.float32)

    def GetTrainFeature(self):
        return torch.tensor(self.train_features.values, dtype=torch.float32)

    def GetTrainLabel(self):
        return torch.tensor(self.train_labels, dtype=torch.float32)

    def GetTestFeature(self):
        return torch.tensor(self.test_features.values, dtype=torch.float32)

