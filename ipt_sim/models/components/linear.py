import torch.nn as nn


class LinearProjection(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.L = nn.Linear(input_size, output_size, bias=False)
        nn.init.kaiming_uniform_(self.L.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        return self.L(x)
