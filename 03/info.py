import torch
import os


if __name__== '__main__':
    # 查看 CUDA 是否可用
    print("CUDA Available: ", torch.cuda.is_available())

    # 查看 GPU 数量
    print("Number of GPUs Available: ", torch.cuda.device_count())

    # 查看每个 GPU 的名称
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: ", torch.cuda.get_device_name(i))

    # 查看当前使用的设备
    # print("Current Device: ", torch.cuda.current_device())

    print("Number of CPUs: ", os.cpu_count())
