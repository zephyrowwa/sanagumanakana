import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
satisfaction_map = {
    'happy': 'satisfied',
    'surprise': 'satisfied',
    'neutral': 'satisfied',
    'sad': 'dissatisfied',
    'angry': 'dissatisfied',
    'disgust': 'dissatisfied',
    'fear': 'dissatisfied'
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def load_model(model_path, device):
    from torchvision.models import convnext_tiny
    model = convnext_tiny(pretrained=False, num_classes=len(emotion_classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def detect_faces(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

def predict_emotion(model, face_img, device):
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    face_tensor = transform(face_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(face_tensor)
        _, pred = torch.max(output, 1)
        emotion = emotion_classes[pred.item()]
        satisfaction = satisfaction_map[emotion]
    return emotion, satisfaction
