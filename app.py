import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion classes (adjust to your model)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Image transforms for ConvNeXt
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model from Hugging Face
@st.cache_resource
def load_model():
    model_url = "https://huggingface.co/your-username/your-model-repo/resolve/main/model.pth"
    response = requests.get(model_url)
    state_dict = torch.load(BytesIO(response.content), map_location='cpu')

    from torchvision.models import convnext_tiny
    model = convnext_tiny(num_classes=len(emotion_labels))
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

st.title("Facial Emotion Recognition")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = image.crop((x, y, x + w, y + h))
        input_tensor = transform(face_img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            label = emotion_labels[pred]

        cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image_cv, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    st.image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)
