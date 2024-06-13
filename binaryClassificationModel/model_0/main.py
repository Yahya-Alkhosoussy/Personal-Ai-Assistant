import sklearn #this is a machine learning library
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import requests
from pathlib import Path


# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

# Checking the samples
# print(f"length of X: {len(X)}", f"\nlength of y: {len(y)}")
# print(f"First 5 samples of X: {X[:5]}")
# print(f"First 5 samples of y: {y[:5]}")

# Creates and plots the datapoints
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "Label": y})

# prints a table with the dataponts
# print(circles.head(10))

# plots the points making the 2 circles for visualization
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu) #Cmap = plt.colormap.RedYellowBlue

# X and Y are of type numpy arrays so we need to convert them to torch tensors


X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

# Split data inton training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, #this is the format for the function
                                                     y,
                                                     test_size=0.2,# 20 percent of data will be test and the rest will be train
                                                     random_state=42)

#Looking at the test and train datasets
# print(f"Length of training set X: {len(X_train)}", f"\nlength of testing set X: {len(X_test)}", f"\nlength of training set y:{len(y_train)}", f"\nlength of testing set y: {len(y_test)}")

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# 1. Construct a model that Subclasses nn.Module
class CircleModelV1(nn.Module):

    def __init__(self):

        super().__init__()

        # 2. Create 2 nn.Linear layers capable of handling the shapes of our data
        self.layer_1 = nn.Linear(in_features = 2, out_features = 5) # takes in 2 features and upscales to 5 features (makes learning faster and more efficient)
        self.layer_2 = nn.Linear(in_features = 5, out_features = 1) # in_features has to match the out_features of a previous layer
        #layer_2 takes 5 in_features from layer_1 and outputs 1 feature (the same shape as y)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layer_2(self.layer_1(x)) # x -> layer_1 -> layer_2 -> output

# 4. Instantiate an instance of our model class and send it to the target device
model_0 = CircleModelV1().to(device)

#replicating the model with nn.sequential()
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

# Making Predictions
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))

# print(f"Length of predictions: {len(untrained_preds)}, shape: {untrained_preds.shape}")
# print(f"Length of test samples: {len(X_test)}, shape: {X_test.shape}")
# print(f"\nFirst 10 predictions:\n {torch.round(untrained_preds[: 10])}")
# print(f"\nFirst 10 labels:\n {y_test[: 10]}")

### Optimizer and loss functions
# Setup the loss function
# loss_fn = nn.BCELoss() # BCELoss = requires inputs to have gone through sigmoid activation function prior to input to BCELoss
# nn.sequential(
#     nn.Sigmoid(),
#     nn.BCELoss()
# ) this is equivalent to the line below, but the code below is more stable numerically
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid activation function built-in

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr = 0.1)

# Calculate accuracy  - out of 100 examples what percentage does out model get right
def accuracy_fn(y_true, y_pred):

    correct = torch.eq(y_true, y_pred).sum().item()

    acc = (correct / len(y_pred)) * 100

    return acc

### Training the model

## 3.1 going from raw logits -> prediction probabilities -> prediction labels

model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device)) [: 5]

y_pred_probs = torch.sigmoid(y_logits)

# Findthe predicted labels
y_preds = torch.round(y_pred_probs)

# In Full (logits -> pred probs -> pred labels)

y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[: 5]))

print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

# get rid of extra dimension
y_preds.squeeze()

## 3.2 making and testing the testing loop

if device == "cuda":

    torch.cuda.manual_seed(42)
else:

    torch.manual_seed(42)

# Set the number of epochs
epochs = 100

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training model
for epoch in range(epochs):

    ### Training

    # Set the model to training mode
    model_0.train()

    # 1. Forward pass
    # Pass the training data through the model and squeeze the output to remove extra dimensions
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # Turn logits -> pred probs -> pred labels

    # Turn logits -> pred probs -> pred labels
    y_pred = torch.round(torch.sigmoid(y_logits))

    # 2. Calculate loss/accuracy
    # Calculate the loss between the model's predictions and the true labels
    loss = loss_fn(y_logits, #nn.BCEWithLogitsLoss expects raw logits as input
                   y_train)

    # Calculate the accuracy of the model's predictions
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    # 3. Optimizer zero grad
    # Clear the gradients of all optimized variables
    optimizer.zero_grad()

    # 4. Loss backward (backpropagation)
    # Compute the gradients of the loss with respect to the model's parameters
    loss.backward()

    # 5. optimizer step(gradient descent)
    # Update the model's parameters using the computed gradients
    optimizer.step()

    ### Testing

    # Set the model to evaluation mode
    model_0.eval()
    with torch.inference_mode():
        #1. Forward pass
        # Pass the testing data through the model and squeeze the output to remove extra dimensions
        test_logits = model_0(X_test).squeeze()

        # Turn logits -> pred probs -> pred labels
        test_pred = torch.round(torch.sigmoid (test_logits))

        # 2. Calculate test loss/acc
        # Calculate the loss between the model's predictions and the true labels
        test_loss = loss_fn(test_logits,
                            y_test)

        # Calculate the accuracy of the model's predictions
        test_acc = accuracy_fn(y_test,
                          test_pred)

    # Print the training and testing metrics every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
        #.#f where # is an integer is how many decimal points is wanted


### 4. making predictions


# Download helpter functions from learn PyTorch repo (if its not already downloaded)
def download_helpers():
    """
    This function checks if a helper_functions.py file exists in the current directory.
    If it does not exist, it downloads the file from the provided URL.

    Parameters:
    None

    Returns:
    None
    """
    # Check if helper_functions.py already exists
    if Path("helper_functions.py").is_file():
        print("helper_functions.py already exists, not downloading")

    else:
        print("downloading helper_functions.py")
        # Send a GET request to the provided URL
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
        
        # Open a new file in write binary mode
        with open("helper_functions.py", "wb") as f:
            # Write the content of the response to the file
            f.write(request.content)


