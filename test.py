import torch
import numpy
a =torch.ones(5)
b=a.numpy()
a.add_(1)
b = b+1
print(b,a)