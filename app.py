import streamlit as st 
import requests
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models


st.title('Image Classifier')

model = models.resnet18(pretrained=True)
model.eval()

LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = [line.strip() for line in requests.get(LABELS_URL).text.splitlines()]

def imagepro(image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()

    ])
    return transform(image).unsqueeze(0)

upload = st.file_uploader('upload an Image',type =['jpg','png','jpeg'])

if upload is not None:
    image=Image.open(upload).convert('RGB')
    st.image(image,caption = 'original Image')

    with st.spinner('Processing....'):
        i_tensor = imagepro(image)
    with torch.no_grad():
        out = model(i_tensor)
        a, predicted = torch.max(out,1)
        label = labels[predicted.item()]

    st.success(f'Predicted Items is:{label}')