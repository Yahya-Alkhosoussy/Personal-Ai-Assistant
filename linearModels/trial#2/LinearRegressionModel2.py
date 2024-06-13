# Creating some data using the linear regression formula of y = weight * X + bias
import torch
from torch import nn

weight = 2.5

bias = 1.5

start = 0

end = 5

step = 0.05

X = torch.arange(start=start, end=end, step=step).unsqueeze(dim=1)


# Linear regression formula
y = weight * X + bias

train_split = int(0.8*len(X))

X_train = X[: train_split]

y_train = y[: train_split]

X_test = X[train_split :]

y_test = y[train_split :]

class LinearRegressionModel2(nn.Module):

    def __init__(self):

        super().__init__()

        # Use nn.Linear() for creating the model paramters (this creates a layer where the parameters lay)
        # behind the scenes (aka linear transform, probing layer, dense layer, fully connected layer)
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.linear_layer(x)
    
    
