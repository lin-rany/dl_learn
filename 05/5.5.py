

import time
import torch


def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


if __name__ == '__main__':
    startTime1 = time.time()
    for i in range(100):
        A = torch.ones(500, 500)
        B = torch.ones(500, 500)
        C = torch.matmul(A, B)
    endTime1 = time.time()

    print(f"try_gpu():{try_gpu()}")

    startTime2 = time.time()
    for i in range(100):
        A = torch.ones(500, 500, device=try_gpu())
        B = torch.ones(500, 500, device=try_gpu())
        C = torch.matmul(A, B)
    endTime2 = time.time()

    print('cpu计算总时长:', round((endTime1 - startTime1) * 1000, 2), 'ms')
    print('gpu计算总时长:', round((endTime2 - startTime2) * 1000, 2), 'ms')
