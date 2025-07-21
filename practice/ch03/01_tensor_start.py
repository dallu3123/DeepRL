import torch
import numpy as np

if __name__ == "__main__":

    a = torch.FloatTensor(3, 2) # 3x2 행렬 생성 값은 0
    print(a)

    b = torch.zeros(3, 4) # 3x4 행렬 생성 값은 0
    print(b)

    print(a.zero_()) 

