from custom_dataset import CustomDataset
import pandas as pd

if __name__ == '__main__':

    dataSet=CustomDataset()

    train_features=dataSet.GetTrainFeature()
    test_features=dataSet.GetTestFeature()

    train_labels=dataSet.GetTrainLabel()

    # 打印出前1个样本及其标签
    for i in range(1):
        train_feature = train_features[i]
        test_feature = test_features[i]
        print('train_feature  shape:', train_feature.shape)
        print('test_feature  shape:', test_feature.shape)
        print('\n\n\n')

        train_label=train_labels[i]
        print('train_label  shape:', train_label.shape)


    # 检查数据集的大小
    print("train_dataset size: ", len(dataSet.GetTrainFeature()))
    print("test_dataset size: ", len(dataSet.GetTestFeature()))
    print("GetTrainLabel size: ", len(dataSet.GetTrainLabel()))