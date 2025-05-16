import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from timm import create_model

# Emotion class labels (adjust if yours differ)
class_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
satisfaction_map = {
    'angry': 'dissatisfied',
    'disgust': 'dissatisfied',
    'fear': 'dissatisfied',
    'sad': 'dissatisfied',
    'happy': 'satisfied',
    'neutral': 'satisfied',
    'surprise': 'satisfied'
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load ConvNeXt model
@st.cache_resource
def load_model():
    model = torch.load("FRconvnext_full(R)(A).pth", map_location='cpu', weights_only=False)
    model.eval()
    return model

# Preprocess uploaded image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(image).unsqueeze(0)

# Predict function
def predict(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
    return class_labels[predicted_class], probs.squeeze().tolist()

# UI
st.set_page_config(page_title="Emotion to Satisfaction", layout="centered")
st.title("Facial Emotion Recognition with Satisfaction Mapping")
st.write("Upload an image to classify emotions using a ConvNeXt model.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No face detected.")
    else:
        model = load_model()

        for (x, y, w, h) in faces:
            face_img = image.crop((x, y, x + w, y + h))
            input_tensor = preprocess_image(face_img)
            label, probabilities = predict(model, input_tensor)
            satisfaction = satisfaction_map[label]

            # Draw result on image
            cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_cv, f"{label} → {satisfaction}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show details
            st.success(f"Emotion: **{label}** → Satisfaction: **{satisfaction}**")
            st.subheader("Confidence Scores")
            st.bar_chart({lbl: prob for lbl, prob in zip(class_labels, probabilities)})

        # Display result
        result_img = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        st.image(result_img, caption="Detected Face(s) with Emotion & Satisfaction", use_column_width=True)
