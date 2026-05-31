## Import Section ##

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models  # This is where all the pretrained models live. ResNet18, VGG, EfficientNet etc. We are using ResNet18 which contains 1k categories of almost 1.2 mill images.
import matplotlib.pyplot as plt  # Trained and the weights are already baked form training.

## Transforms - Resize image to ResNet18 224x224 ##

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),  # Converts the image from PIL/numpy format to a PyTorch tensor & also scales pixel values down from 0-255 to 0-1.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # mean is avg brightness of RGB across images. #std is the spread of RGB across image aka how bright the images are.
])  # applies ImageNet mean and std ^^. ImageNet is the database that microsoft uses to train ResNet18.

## Dataset - Using CIFAR-10 a famous dataset w/ 60k images across 10 categories - plane, car, birds etc. ##

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
# root will create a folder in our current directory called data. train=... means if True download the the training set(50k). False means download the test dataset(10k)
# download=True if dataset isnt there then download it. If it is there skip.
# transform=None means do not apply the transforms we defined above includes resize, toTensor, normalize to every image because we filter the image first.
# left transform is the variable CIFAR expects and none is we dont want to apply yet because filtering is the next set. Applying to 50k images requires alot of power.


## Filter out Datset - From CIFAR-10 we only need Planes (0) and Cars(1) only ##

def filter_classes(dataset, classes):  # fxn that takes in dataset and a list of classes we want. our dataset is CIFAR-10. Classes [0, 1](planes and cars)
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    # i for i loop through each index. enumerate gives pairs for e.g (0, plane),(1,car) etc. (_, label) means each item is (image, label) ignore image with '_'. We only need to sort the image by label not open each folder.
    return torch.utils.data.Subset(dataset, indices)  # build a smaller dataset aka subset which will give us these specific indices in our case: planes and cars.


train_dataset = filter_classes(train_dataset, [0, 1])  # filter classes by plane and cars.
test_dataset = filter_classes(test_dataset, [0, 1])

## Apply Transforms - Data has been filtered & now apply transform ##


train_dataset.dataset.transform = transform  # train_dataset is our subset. a list of index numbers. doesnt hold the act images. the real images are in .dataset.
test_dataset.dataset.transform = transform

# '.dataset' -> go into our dataset. '.transform' empty slot where processing instructions are supposed to be. =transform fill the slot with our transform recipe in line 12.

## DataLoaders ##
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,  # we chose 32 over 64 because ResNet18 images are 224x224 vs MNIST 28x28. bigger images=more memory= smaller batch size.
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False  # because we are testing we want consistent results. in train we said true because we want to prevent overfitting.
)
# Our flow today is ResNet18(the neural architecture) learned from ImageNet(contains 1.2mill images & 1k categories) -> We take the predetermined model aka pretrained ResNet18 -> Fine tune it on CIFAR-10 -> It learns planes vs cars.


## Loading ResNet18 ##

# 'models.' is the torchvision lib we imported. 'resnet18' the architecture we want. 'pretrained=True' give me the pretrained version already trained on ImageNet.
model = models.resnet18(pretrained=True)

for weights in model.parameters():  # for each weight in model.parameters. next line means freeze it.
    weights.requires_grad = False  # '.requires_grad' mean dont calculate gradients for this layer, so backpropagation skips. The weight stays frozen.


## Swapping the last layer ##

model.fc = nn.Linear(512, 2)  # replace 512 which is the last layer size of ResNet with 2 which is our plane or car


## Loss fxn & optimizer | Training Loop ##

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


## Accuracy Section ##

model.eval()  # switches off training mode and turns on eval mode.
correct = 0  # 2 counters starts at zero
total = 0

with torch.no_grad():  # turns off gradient tracking.
    for images, labels in test_loader:  # loops through test data, 32 images at a time.
        outputs = model(images)  # feed images through the model gets 2 scores per image.
        _, predicted = torch.max(outputs, 1)  # look at 2 scores and pick the highest one. '_' ignore the score value. predicted means which class one(0=plane or 1=car)
        total += labels.size(0)  # Add 32 to total each batch. labels.size(0) means how many labels in the batch which is 32.
        correct += (predicted == labels).sum().item()  # compare predictions to real labels. .sum() count the trues..item() convert to plain no. add to counter.

    print(f"Accuracy: {100 * correct / total:.2f}%")  # will calculate the accuracy percentage -> (correct/total)x100 = accuracy %

torch.save(model.state_dict(), 'resnet18_planes_cars.pth')  # This will save the weights for the next day.
# torch.save -> PyTorch's save fxn
# mode.state_dict() -> extracts the learned weights and biases form the model. Basically everything the model learnt during training.
# .pth is the standard PyTorch extension for saving weights.
