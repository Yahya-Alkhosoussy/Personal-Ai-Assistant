import torch

class LinearRegressionModel(torch.nn.Module): # <- almost everything in PyTorch inhertis from nn.Module

    def __init__(self):

        super().__init__()

        self.weights = torch.nn.Parameter(torch.randn(1, # <- start with a random weight and try to adjust to the ideal weight.
                                               requires_grad=True,
                                               dtype=torch.float))
        self.bias = torch.nn.Parameter(torch.randn(1,
                                            requires_grad=True,
                                            dtype=torch.float))

    # forward() method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data

        return self.weights * x + self.bias #this is the linear regression formula