## Application Programming Interface aka API ##
# Our Previous models was chefs who had the ability to cook amazing food
# FastAPI is the restaurant that lets customers order from that chef.
# We will wrap our model in a API, so

## Import Section ##

from fastapi import FastAPI, UploadFile, File  # fastapi(library) - Fastapi is the webframework. Uploadfile(handles files upload) and File(tells FastAPI to expect a file in the req.)
from PIL import Image  # PIL opens and processes images.
import torch  # Main PyTorch framework
import torch.nn as nn  # Contains the neural Network tools
import torchvision.transforms as transforms  # Contains the tools to transform and preprocess images before feeding them into the network.
import torchvision.models as models  # All the pre trained models live here. ResNet18, VGG etc.
import io  # handles image bytes from upload


## Create the FastAPI ##

app = FastAPI()

## Transforms ##

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),  # Converts image from PIL to PyTorch Tensor & also scales pixel down to between 0-1.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # mean -> avg brightness of RGB across images. std ->how differ are pixels across image form avg.
])


## Load Model ##

model = models.resnet18(weights=None)  # model is not pretrained aka empty
model.fc = nn.Linear(512, 2)  # replaces 512 which is the size of the last layer in ResNet with 2 which is a plane or car.
model.load_state_dict(torch.load('resnet18_planes_cars.pth', map_location='cpu'))  # 'model.load_state_dict 'load weights into model structure. 'torch.load()' PyTorch's build in load fxn reads the .pth file. 'map_location='cpu' load's weight onto the CPU.
model.eval()  # turns off training and turns on evaluation mode.#

## Class Names ##

classes = ['plane', 'car']  # create 2 classes plane and car

## API Endpoint ##


@app.post("/predict")  # '@app.post' is the post request aka acceps incoming mail. '/predict' the url endpoint aka our address on the mailbox.
async def predict(file: UploadFile = File(...)):  # 'async' means can handle multiple requests at once. defining predict our fxn. 'file : UploadFile' expects a file upload.
    image_data = await file.read()  # await means wait for the file to fully upload. file.read() read the raw bytes of the image.
    image = Image.open(io.BytesIO(image_data)).convert("RGB")  # 'Image.open()' open it as a PIL image. 'io.BytesIO(image_data)' convert raw bytes into file like object. '.convert("RGB") make sure it is RGB not greyscale.
    image = transform(image).unsqueeze(0)  # 'transform(image)' apply our resize, ToTensor, normalize. 'unsqueeze(0)' adds batch dimensions - shape goes from [3,224,224] to [1,3,224,224]. 1 signifies that we have on image in our batch, 3 is our RGB aka color channel, 224x224 pxls.

    with torch.no_grad():  # turn off gradient tracking aka we just want predicting.
        outputs = model(image)  # feed image through ResNet18 and get 2 scores.
        _, predicted = torch.max(outputs, 1)  # Pick the highest score. predicted will be 0(plane) or 1(car).  torch.max(outputs,1) returns 2 things - 1st actual score value we dont need this so _ & index of highest score this is our predicted.

    return {"prediction": classes[predicted.item()]}  # Returns JSON like {"prediction" : "plane"} for e.g if classes[0] then "plane" if classes[1] then "car"


## Run the API ##

if __name__ == "__main__":
    import uvicorn  # the server that runs FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)  # uvicorn.run -> starts the server | app is our FastAPI we created w/ app = FastAPI() | host="0.0.0.0" accepts request from anywhere | port=8000 means think of ports as a apartments in a apt building that do not exist. Port numbers are stored as 16bits integers. each bit can only be 0 or 1. so 2 total values 2^16 is 65535 so our port has to be between 0 to 65535 because we count from zero not 1 or it would be 65536.
