## Import Section ##

import torch  # Main PyTorch framework
import torchvision  # Datasets and pretrained models
import torch.nn as nn  # Neural Network Tools
import torchvision.transforms as transforms  # Image preprocessing tools
import matplotlib.pyplot as plt  # Plotting

## Transforms - preprocessing images ##

transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image from PIL to PyTorch Tensor format & scales value to between 0-1.
    transforms.Normalize((0.5,), (0.5,))  # Normalize to -1 to 1
])


## Load Dataset - downloading and Loading FashionMNIST ##

train_data = torchvision.datasets.FashionMNIST(
    root="./data",  # Save to data folder
    train=True,  # Download training set(60k images)
    download=True,  # Download if not already downloaded.
    transform=transform  # Apply our Transforms.
)

test_data = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,  # Download our test set(10k images)
    download=True,
    transform=transform
)

## Dataset Loader - Feeds data to the network in batches ##

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=64,
    shuffle=True  # To prevent overfitting while training
)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)

## Neural Network ##


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # 1 input channel(greyscale), 32 filter, 3x3 kernels
            nn.ReLU(),  # Activation fxn
            nn.MaxPool2d(2)  # Halves the image size, keeps the strongest features.
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 13 * 13, 128),  # (32 * 13 * 13 features going in) -> 128 features going out aka 128 neurons.
            nn.ReLU(),
            nn.Linear(128, 10)  # 128 neurons and 10 clothing categories.
        )

    def forward(self, x):
        x = self.conv_layers(x)  # Pass it thru the conv layers.
        x = x.view(-1, 32 * 13 * 13)  # Flatten for fc layers.
        x = self.fc_layers(x)  # Pass thru the layers.
        return x


model = CNN()  # Create Instance for CNN

## Loss function & Optimizer ##

loss_fn = nn.CrossEntropyLoss()  # Classification Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam adjusts weights. , lr= 0.001

## Training Loop ##

epochs = 5

for epoch in range(epochs):
    model.train()  # Tells PyTorch we are in training mode.
    for images, labels in train_loader:
        outputs = model(images)  # Forward pass - get prediction
        loss = loss_fn(outputs, labels)  # Calculate Loss

        optimizer.zero_grad()  # Clear old gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights.

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


## Testing Section ##

correct = 0
total = 0

model.eval()  # Turn on eval mode

with torch.no_grad():  # no gradient tracking needed.
    for images, labels in test_loader:
        outputs = model(images)  # get predictions
        _, predicted = torch.max(outputs, 1)  # pick highest score.
        total += labels.size(0)  # add batch size to total count
        correct += (predicted == labels).sum().item()  # count correct

    print(f"Accuracy: {100 * correct / total:.2f}%")

## Save Model ##

torch.save(model.state_dict(), 'fashion_cnn.pth')  # Save weights to pth file.
