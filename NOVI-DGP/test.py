import torch
from sklearn.datasets import load_boston

boston = load_boston()
X = boston.data
y = boston.target
print(X)
print(y)
