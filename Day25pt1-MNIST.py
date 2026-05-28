## Input Section ##

import torch   # Main PyTorch framework
import torch.nn as nn   # Neural network tools
import torchvision     # Image datasets & tools.
import torchvision.transforms as transforms  # Tools to transform and preprocess images before feeding them to the network.

## Transforms - preprocessing images ##

transform = transforms.Compose([  # Chains multiple transforms together. Does them in order one after another.
    transforms.ToTensor(),        # Converts the image from PIL format into a PyTorch tensor. Also automatically scales pixel values from 0-255 to 0-1.
    transforms.Normalize((0.5,), (0.5,))  # Takes the values of 0-1  and makes them -1 to +1. First bracket (0.5,) means - Subtract 0.5 from every pxl.
])   # Second bracket (0.5,) means stndrd deviation aka divide it by 0.5. The reason why we need -1 is because we need e.g fully working compass not half.

## Load Dataset - downloading and loading MNIST ##

train_data = torchvision.datasets.MNIST(   # MNIST is a famous dataset of 70k handwritten digit images that torchvision has built into it
    root="./data",  # Creates a data folder in your current directory and saves the dataset there.
    train=True,     # Gives you the training set of 60k images. If you set this as False you get the dataset of 10k images.
    download=True,  # If the dataset is not on your computer then download it automatically.
    transform=transform  # Apply the transforms we defined to tensor and normalize to every image you load.(line 8 and 9)
)

## Data Loader - Feeds data to the network in batches ##

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=64,   # Batch size is 64 each.
    shuffle=True     # Shuffles our data randomly.
)

## Neural Network ##


class NeuralNetwork(nn.Module):   # nn.Module is the blueprint book and contains the exact blueprint NeuralNetwork.
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):    # self is the object we created which is model. x is our input data which is our batches each being fed each iteration.
        return self.layers(x)


model = NeuralNetwork()  # Created object NeuralNetwork() stored in model. Blueprint is class NeuralNetwork(). Blueprint book is nn.Module


## Loss function and Optimizer ##

loss_fn = nn.CrossEntropyLoss()  # measures how wrong the prediction is compared to the correct answer. nn is the module we imported.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # torch.optim is pytorch's optimizer variable contains optimizers like adam that we are using.
# () creates the object and contains our learning rate and model.parameters tells adam which weights to update it returns all weights and biases in the
# network. In summary line says - Create an adam optimizer that will update all weights in our models with a learning rate of 0.001.


## Training Loop ##

epochs = 5  # One epoch = network has seen all 60k images once.

for epoch in range(epochs):  # outer loop runs 5 times
    for images, labels in train_loader:  # inner loop each iteration grabs one batch of 64 images and their correct labels(0-9). train loader handles this.
        images = images.view(-1, 784)  # reshapes each image from 28x28 grid into a flat list of 784 numbers. Network requires flat input. -1 mean figure
        # out batch size automatically.

        outputs = model(images)  # feeds the images through the network. Gets back 10 numbers per image. the network's prediction is from 0-9 digits.
        loss = loss_fn(outputs, labels)  # compares the networks predictions to the correct answer. High no = very wrong and vice versa.

        optimizer.zero_grad()  # clears the gradients from the previous batches. Needs to be done or gradients will overlap and pile up.
        loss.backward()  # backpropagation sends the error backwards through the network. calculating how much each weight contributed to the mistake.
        optimizer.step()  # updates the weights using the gradients calculated above by loss.backward(). loss.backward() is gradient descent happening.

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")  # after each epoch prints the current loss. goes down each epoch because network is learning.


## Testing Section ##

correct = 0   # counter for correct predictions
total = 0     # counter for total images seen, starts at 0

with torch.no_grad():     # turns of gradient tracking. We are testing not training. Saves memory and runs faster.
    for images, labels in train_loader:  # loops through all images and their labels.
        images = images.view(-1, 784)    # same as training - flatten 28x28 image to 784
        outputs = model(images)  # feed images through the network & gets 10 numbers each in a row. Each number is out of 1 representing confidence for each number it thinks it is.
        _, predicted = torch.max(outputs, 1)  # finds the highest number against the 10 outputs for each image. highest confidence will tell us which no. is right.
        total += labels.size(0)  # adds 64 to each batch size. zero means dimension zero = how many images. dimension 1 = how many pixels.
        correct += (predicted == labels).sum().item()  # compares predictions to the correct answers
# above line basically says count how many predictions matches the correct labelsand then add it to the correct counter.
# prediced == labels means it compares the prediction to the correct answer. .sum(0) counts how many trues there are. true=1, false = 0 | .item()
# converts the pytorch tensor number into a regular python int so we can add it into correct.

print(f" Accuracy: {100 * correct / total:.2f}%")  # calculates accuracy as a percentage and prints it.

# Any accuracy above 95% is considered excellent!

## Summary of classes we used ##

# nn.Module -> base class for all networks
# nn.Linear -> a fully connected layer
# nn.ReLU -> activation fxn
# nn.Sequential -> chains layers together
# nn.CrossEntropyLoss -> loss function

## Summary of the program ##

# Feed images in
# Get predictions
# Calculate Loss
# Backpropagate
# Update Weights
# Repeat
