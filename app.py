import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from load_model import load_model

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

# Load model
model = load_model('FRconvnext_full(R)(A).pth')
device = torch.device('cpu')

# Define your emotion classes
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# App UI
st.title("Facial Emotion Recognition App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = img_np[y:y+h, x:x+w]
        face_pil = Image.fromarray(face).resize((224, 224))
        input_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            emotion = classes[pred]

        cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_np, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    st.image(img_np, caption="Detected Emotions", use_column_width=True)
