from custom_dataset import CustomDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
if __name__ == '__main__':
    custom_dataset = CustomDataset()
    # 获取特征和标签
    features = custom_dataset.GetTrainFeature()
    labels = custom_dataset.GetTrainLabel()

    # 分割数据集为训练集和测试集
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,
                                                                                random_state=42)
    features_train = np.array(features_train)
    features_test = np.array(features_test)
    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)
    # 创建随机森林回归模型
    regressor = RandomForestRegressor(random_state=0)

    # 定义要进行网格搜索的超参数字典
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40]
    }

    # 创建 GridSearchCV 对象
    grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5)

    # 执行网格搜索
    grid_search.fit(features, labels.ravel())

    # 打印找到的最佳超参数
    print("Best parameters: ", grid_search.best_params_)

    # 根据找到的最佳超参数，建立并训练模型
    best_regressor = RandomForestRegressor(n_estimators=grid_search.best_params_['n_estimators'],
                                           max_depth=grid_search.best_params_['max_depth'],
                                           random_state=0)
    best_regressor.fit(features, labels.ravel())

    predictions=best_regressor.predict(features_test)
    # 计算预测结果的均方误差
    mse = mean_squared_error(labels_test, predictions)
    print("Mean Squared Error: ", mse)