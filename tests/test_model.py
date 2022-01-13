import pytest
import torch

from src.models.model import MyAwesomeModel

b = 16
x = torch.rand(b, 1, 784)
model = MyAwesomeModel()
y = model(x)

assert y.shape[0] == b, "Batch size should not change throughout forward"
assert y.shape[1] == 10, "Number of output classes should be 10"
