# BinaryClassificationModel

## Imports

1. `import torch` this import is the backbone of the code
2. `from torch import nn` this import ensures that the neural network is imported from PyTorch
3. `import requests` this import ensures that if we need to download a code file from the internet like **helper_functions.py** then it is possible to do so
4. `from pathlib import Path` creates a path for all the downloaded files from the internet or any files like a text file that documents the progress of the Model results
5. `from helper_functions import *` imports everythign from **helper_functions.py** to help us plot everything
6. `import sklearn` this import is to help give us a dataset
7. `from sklearn.datasets import make_circles` this import gives us the make_circles function from sklearn.datasets which helps us in making a dataset for the binary classification model
8. `import pandas as pd` this import makes a dataframe for the binary classification model
9. `import matplotlib.pyplot as plt` this import makes it possible for the datasets to be plotted which is important for **visualization** 

## 1. Making some data

`make_circles()` this creates a dataset and takes the following parameters: 
* `n_samples`: number of samples whcih is an int or tuple of shape (2,). *In this case I chose an int of 1000*
* `shuffle`: Whether to shuffle the samples. *boolean, default=True.*
* `noise`: Standard deviation of Gaussian noise added to the data. *float, default=None.* **I chose a noise of 0.3** 
more on Gaussian noise: https://en.wikipedia.org/wiki/Gaussian_noise#:~:text=In%20signal%20processing%20theory%2C%20Gaussian,known%20as%20the%20Gaussian%20distribution).
* `random_state`: Determines random number generation for dataset shuffling and noise. **chosen value is 42** *default=None*

`make_circles()` returns the following values:
1. X: numpy array of shape (n_samples, 2) and is the generated dataset
2. y: numpy array of shape(n_samples,) and is the integer labels (0 or 1) for class membership of each sample (used for binary classification)

**Note**: The data we're working with is often refered to as a toy dataset, a dataset which is small enough to experiment on but still sizeable enough to practice fundementals

## 2. Making the model:

Let's build a model to classify our blue and red dots

To do so, it is needed to:

1. Setup device agnostic code so our code will run on an accelerator (GPU) if there is one
2. Construct a model(by subclassing `nn.Module`)
3. Define a loss function and optimizer
4. Create a training and test loop

### 2.1 Making the model

After setting up device agnostic code it is necessary to make a model that:

1. Subclasses `nn.Module` (almost all models in PyTorch subclass `nn.Module`)
2. Create 2 `nn.Linear()` layers that are capable of handling the shapes of our data
3. Defines a `forward()` method that outlines the forward pass (or forward computation) of the model
4. Instantiate an instance of our model class and send it to the target device

look at the code after #2 and before #2.2

### 2.2 Creating the optimizer and loss function

Which loss function or optimizer should you use for a classification model?

Again.. this is problem specific

For example for regression (picking or predicting a number) you might want MAE or MSE (mean absolute error or mean squared error).

For classification you might want binary cross entropy or categorical cross entropy (cross entropy)

As a reminder, a loss function finds out how wrong your model is.

And for optimizers, two of the most common and useful are SGD and Adam, however PyTorch has many built-in options

* For some common choices of loss functions and optimizers - https://www.learnpytorch.io/02_pytorch_classification/

* For the loss function we're going to use torch.nn.BCEWithLogitsLoss() for more on wwhat binary cross entropy (BCE) is, check out this article - https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a

* For definition of what a Logit is in deep learning - https://stackoverflow.com/a/52111173/24839721

* For different optimizers see torch.optim

* Sigmoid activation function - https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

* the loss function of choice is `nn.BCEWithLogitsLoss()` which has the sigmoid activation function built in and expects a logit as an argument instead of a prediction probabiltiy.

## 3. Training model

To train our model, we're going to build a training loop, with the following steps:

1. forward pass
2. Calculate the loss
3. Optimizer zero grad
4. loss backward (backpropogation)
5. optimizer step (gradient descent)

### 3.1 Going from raw logits -> prediction probabilities -> prdiction labels

Our model outputs are going to be raw **logits.**

We can convert these **logits** into probabilities by passing them through some kind of activation function (eg. sigmoid for binary classification and softmax for multiclass classification).

Then we can convert our model's prediction probabilities to prediction labels by either rounding them or taking the `argmax()`

For our prediction probability values, we need to preform a range-style rounding on them:

`y_pred_probs` >= 0.5, `y=1` (class 1)

`y_pred_probs` < 0.5, `y=0` (class 0)

### 3.2 Building a training and testing loops

standard training and testing loops where `epochs` are the number of repitions done in the training loop and as stated in #3 the steps to train the model is:

1. forward pass
2. Calculate the loss
3. Optimizer zero grad
4. loss backward (backpropogation)
5. optimizer step (gradient descent)

* `y_pred = torch.round(torch.sigmoid(y_logits))`: Turns logits -> pred probs -> pred labels

## 4. Make predictions and evaluate the model

from the metrics it looks like our model isn't learning anything..

So To inspect it let's make some predictions

visualize the data

To do so, we're going to import a function called `plot_decision_boundary()`

to get the function we needed to import a file from github called `helper_functions.py` and how it is done is found in the following steps:

1. importing `requests` as it is the main way to download raw files from the internet
2. importig `Path` as it creates or makes sure the file exists and is accessible 
3. checking if the file exists or not using an if statement
4. if it does not exist then using `requests.get(link)` where link is the link to the raw file, it will download the file and have access to its contents
5. using `with open("helper_functions.py", "wb") as file` we are able to write the contents of `requests.get(link)` and write them onto the new file "helper_functions.py"
6. import the functions from the file "helper_functions.py" using `from helper_functions import *` to import all the functions in the file.

## 5. Improving a model (from a model perspective)

A good way to improve the model is by:
* Adding more layers - give the model more chances to learn about patterns in the data
* Add more hidden unites - go from 5 hidden unites to 10 hidden unites