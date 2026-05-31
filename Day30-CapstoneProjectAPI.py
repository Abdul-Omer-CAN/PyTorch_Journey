## Import Section ##

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import io

## Create the FastAPI ##

app = FastAPI()

## Transforms ##

transform = transforms.Compose([
    transforms.Resize((28, 28)),  # will resize to 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

##  CNN Architecture from CapstoneProjectTrain.py File ##


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


## Loading Model ##

model = CNN()  # Create Instance for CNN
model.load_state_dict(torch.load('fashion_cnn.pth', map_location='cpu'))
model.eval()

## Class names ##

classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

## API Endpoint ##


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("L")  # L for greyscale image our model is built on FashionMNIST. They are greyscale. black and white only contains one channel. Because of this nn.Conv2d(1, 32...) that means 1 input channel(greyscale). RGB would give 3 channels and our program would crash.
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return {"prediction": classes[predicted.item()]}

## Run the API ##

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500)
