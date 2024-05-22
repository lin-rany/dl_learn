import os
import pandas as pd
import tensorflow as tf

def create_csv():
    os.makedirs(os.path.join('.', 'data'), exist_ok=True)
    data_file = os.path.join('.', 'data', 'house_tiny.csv')
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')  # 列名
        f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')


def read_csv():
    data_file = os.path.join('.', 'data', 'house_tiny.csv')
    data = pd.read_csv(data_file)
    print(f"data:\n{str(data)}")
    return data


def fill_data():
    data = read_csv()
    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    print(f"inputs:\n{str(inputs)}\noutputs:\n{str(outputs)}")

    newinputs = pd.get_dummies(inputs, dummy_na=True)
    print(f"newinputs:\n{str(newinputs)}")
    X = tf.constant(newinputs.to_numpy(dtype=float))
    y = tf.constant(outputs.to_numpy(dtype=float))
    print(f"X:\n{str(X)}\ny:\n{str(y)}")


if __name__ == '__main__':
    fill_data()
