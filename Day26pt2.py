## Import section ##

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

## Transforms - preprocessing images ##

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

## Load Dataset & downloading and loading MNIST ##

train_data = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

## Data Loader - Feeds data to the batches in the network ##

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=64,  # Batch size is 64
    shuffle=True    # Shuffles our data randomly
)

## CNN aka Convolutional Neural Network ##


class CNN(nn.Module):   # nn.Module is the blueprint book and contains the blueprint of CNN inside it.
    def __init__(self):  # The constructor - runs when you create a CNN object.
        super().__init__()  # Calls on nn.Module's constructor first. Setups the PyTorch's internal machinery. Without this the network wont run.
        self.conv_layers = nn.Sequential(  # Creates the convolutional section stores it in 'self.conv_layers'.
            nn.Conv2d(1, 32, kernel_size=3),  # creates 32 different 3x3 kernels across the image. Each kernel learns to detect a different feature e.g curve, edges.
            nn.ReLU(),  # Activation fxn. Negative = 0. Positive stays the same. Applied after convolution.
            nn.MaxPool2d(2)  # Pooling layer. Takes every 2x2 patch. Keeps only the max value. Reduces our 26x26 output to 13x13. Keeps the strongest features.
        )
        self.fc_layers = nn.Sequential(  # Fully connected layers. The classifier part. Takes the features extracted from conv_layers and decides what digit it is.
            nn.Linear(32 * 13 * 13, 128),  # 32 features maps each 13x13=5408 numbers flattened into one vector. connects to 128 neurons.
            nn.ReLU(),  # Activation again.
            nn.Linear(128, 10)  # 128 neurons -> 10 outputs. One per digit 0-9.
        )

    def forward(self, x):
        x = self.conv_layers(x)  # Passes the image thru the convolutional section -> ConV2D -> ReLU -> MaxPool2d = 13 x 13
        x = x.view(-1, 32 * 13 * 13)  # '-1' means figure out the batch size automatically. We need to flatten from 3d to 1d after conv_layers by multiplying 32*13*13=5408 numbers.
        x = self.fc_layers(x)  # passes the flattened vector through the fully connected layers. Linear -> ReLU -> Linear. Takes the 5408 features and classifies into 10 digit outputs.
        return x  # Returns the final 10 numbers. One per digit. The highest no. The network's prediction.


model = CNN()

## Loss function and Optimizer ##

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

## Training Loop ##

epochs = 5

for epoch in range(epochs):
    for images, labels in train_loader:

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

## Testing Section ##

correct = 0
total = 0

with torch.no_grad():
    for images, labels in train_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")


# NOTE-> Our initial input is 28x28 image. our Kernal is 3x3. So it cant go all the way to the edge. 1 pixel is lost one each side.Memorize this formula
# [(input - kernel) / stride] + 1 -> in our case [(28 - 3) / 1] + 1 = 26x26
# After we get 26x26 MaxPool2d in line 41 takes our 26x26 and divides it by 2 -> 13x13
# To summarize it starts at 28x28 | After Conv2d = 26x26 | After MaxPool2d = 13x13
# Difference between CNN and Regular neural network is 'images = images.view(-1, 784)' NOT REQUIRED in Training Loop for CNN. Flattening happens in foward.
