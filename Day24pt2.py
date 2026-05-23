import torch
import torch.nn as nn       # Is the neural network model that contains all the layers and tools.

# In PyTorch we create a neural network by making a class that inherits from nn.Module


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()  # super() here calls for the parents class of mm.Module and runs its initiation(setup) without the neural network wont function.
        self.layers = nn.Sequential(
            nn.Linear(784, 128),  # takes 784 inputs and connects them to 128 neurons in a hidden layer. Each connection has its own weight and bias. So nn.Linear is one fully connected layer.
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):   # x is our input data, self refers to the object we created which is model.
        return self.layers(x)


model = NeuralNetwork()  # the object created from our blueprint which is 'class NeutralNetwork(nn.Module)'.
print(model)

x = torch.randn(1, 784)  # Creates a tensor w/ 1 sample & 784 random numbers aka pixels in our case. Tensor is a multi dimensional array. Like numPy arrays.
output = model(x)
print(output)

# We are going to build a neural network of: (lines 7 to 13 above)
# Input layer of 784 neurons
# Hidden layer of 128 neurons
# Output layer of 10 neurons(0-9 digits.)

# ReLU is a activation function. If number is negative makes it zero. If positive is keeps it as is. max(0, x). It just takes the max between 0 and x.
# If anything negative then it becomes zero. It picks the maximum between the two numbers. the reason why we left the brackets empty is because this is
# the default.
# Sigmoid squished everything between 0 and 1.
# ReLU is faster than sigmoid and works better in deeper networks. Most modern neural networks use ReLU instead of Sigmoid.
# Each nn.Layer is one connection between two layers. and nn.sequential runs it from top to bottom order.
# PyTorch - torch.randn(784) unlike NumPy arrays - np.random.randn(784) can run on GPU which is alot faster.
# An array is a container that holds a collection of values(same type) stored together. You can access them by index as we know. Each value is
# called an element.
