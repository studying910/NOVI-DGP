#!/usr/bin/env python3

import torch

from .mean import Mean


class LinearMean(Mean):
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None

    def forward(self, x):  # matrix multiplication: x * self.weights: (n,d)*(d,1)=(n,1)
        res = x.matmul(self.weights).squeeze(-1)  # squeeze the '1' in feature dim to form the same size as input
        if self.bias is not None:
            res = res + self.bias  # broadcast addition
        return res
