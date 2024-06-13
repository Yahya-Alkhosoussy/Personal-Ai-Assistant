import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path
from LinearRegressionModel2 import *

def plot_prediction(train_data = X_train,
                    train_labels = y_train,
                    testing_data = X_test,
                    testing_labels = y_test,
                    prediction = None):
    
    plt.figure(figsize=(10, 7))

    # Plot training dara in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label= "Training data")

    # Plot test data in green
    plt.scatter(testing_data, testing_labels, c="g", s=4, label="Testing data")

    # checking if there are predictions
    if prediction is not None:
        # Plot the prediction
        plt.scatter(testing_data, prediction, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})

    # show the plot
    plt.show()

# plot_prediction()

torch.manual_seed(42)

model_1 = LinearRegressionModel2()

# Setup the loss function
loss_fn = nn.L1Loss() # same as MAE

# Setup our optimizers
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.01)


# Training loop
epochs = 200


for epoch in range(epochs):
    model_1.train()

    # 1. Forward pass
    y_pred = model_1(X_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer Zero Grad
    optimizer.zero_grad()

    # 4. Perform backprogation
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing 
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)

        test_loss = loss_fn(test_pred, y_test)

    # Print out what is happening
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")


# Making and evaluating predictions
        
# Turn the model into evaluation mode
model_1.eval()

with torch.inference_mode():
    y_preds = model_1(X_test)
    plot_prediction(prediction=y_preds)


# Saving the model

# 1. Making the model directory
MODEL_PATH = Path("Models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Making the model save path
MODEL_NAME = "Putting_Everything_Together_Model_1.pth"
MODEL_SAVE_PATH =  MODEL_PATH / MODEL_NAME

# 3. Saving the models dictionary (state_dict()) using torch.save 
print(f"Saving the model to path: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(),
            f=MODEL_SAVE_PATH)

# Check in new file