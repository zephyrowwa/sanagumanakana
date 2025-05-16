import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from timm import create_model

# Emotion class labels (adjust if yours differ)
class_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

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
st.set_page_config(page_title="Facial Emotion Recognition - ConvNeXt", layout="centered")
st.title("Facial Emotion Recognition")
st.write("Upload an image to classify emotions using a ConvNeXt model.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model = load_model()
    input_tensor = preprocess_image(image)
    label, probabilities = predict(model, input_tensor)

    st.success(f"Predicted Emotion: **{label}**")

    st.subheader("Confidence Scores")
    prob_dict = {label: prob for label, prob in zip(class_labels, probabilities)}
    st.bar_chart(prob_dict)
