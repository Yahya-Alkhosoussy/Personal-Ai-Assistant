import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from LinearRegressionModel import LinearRegressionModel

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# Create a train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[: train_split], y[: train_split]
X_test, y_test = X[train_split :], y[train_split :]

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    plots training data, test data and compares predictions
    """
    plt.figure(figsize=(10,7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    #plot testing data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")

    # Are there predictions?

    if predictions is not None:
        # Plot the predictions if they exist
        plt.scatter(test_data, predictions, c="r", s=4, label="Prediction")

    # Show the legend
    plt.legend(prop={"size":14})

    plt.show()

# plot_predictions()

# Create a linear regression model class

    
# Create a random seed to get the same values every time
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module)
model_0 = LinearRegressionModel()

## Check out the parameters
# print(list(model_0.parameters()))

## Better way
# print(model_0.state_dict())

# make predictions with model
with torch.inference_mode(): # inference = predictions
  y_preds = model_0(X_test)

# print(y_preds)

# print("\n")

# print(y_test)

# plot_predictions(predictions=y_preds)
  
# Setup a loss function
loss_fn = torch.nn.L1Loss()

# Setup an optimizer (stochastic gradient descent)
optimizer = torch.optim.SGD(params=model_0.parameters(), # we want to optimize the parameters present in our model
                                               lr=0.01) # lr = learning rate = possibly the most important hyperparameter you can set

# An epoch is one loop through the data... (this is a hyperparameter because we've set them ourselves)
epochs = 200

## tracking progress and different values
epoch_count = []
loss_values = []
test_loss_values = []

### Training
# 0. Loop through the data
for epoch in range(epochs):
    # Set the model to training mode
    model_0.train() # train mode in PyTorch that require gradients to require gradients

    # 1. forward pass
    y_pred = model_0(X_train)

    # 2. Calculate the loss
    loss = loss_fn (y_pred, y_train)

    # if epoch % 1000 == 1:
    #     print(f"loss = {loss}")

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. preform backpropagation on the loss with respect to the parameters of the model
    loss.backward()

    # 5. step the optimizer (preform gradient descent)

    # by default how the optimizer changes wo;; acculumate through the loop so we have to zero them above in step 3 for the next 
    # iteration of the loop
    optimizer.step()


    ### Testing
    
    # turns off different settings in the model not needed for the evaluation
    model_0.eval() 

    # turns off the gradient tracking, in older pytorch code torch.no_grad() may be used
    with torch.inference_mode(): 
        # 1. do forward pass:
        test_pred = model_0(X_test)

        # 2. calculate the loss
        test_loss = loss_fn(test_pred, y_test)
    
        # Print out what is happening
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            loss_values.append(loss)
            test_loss_values.append(test_loss)
            print(f"Epoch: {epoch} | Test: {loss} | Test loss: {test_loss}")


# Print out model state_dict()
print(model_0.state_dict())

with torch.inference_mode():
    y_pred_new = model_0(X_test)


# Plotting the loss curves
plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Train Loss")
plt.plot(epoch_count, np.array(torch.tensor(test_loss_values).numpy()), label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
# plt.show()

# plot_predictions(predictions=y_pred_new)

# Saves progress to a text file
with open('progress_and_values.txt', 'w') as a:
    for i in range(len(epoch_count)):
        
        a.write(f"Epoch count: {epoch_count[i]} \n")
        a.write(f"Loss values: {loss_values[i]} \n")
        a.write(f"Test Loss Values: {test_loss_values[i]} \n")
        a.write()


