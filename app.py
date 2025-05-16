import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from timm import create_model

# Emotion class labels (adjust if yours differ)
class_labels = ['angry', 'disgusted', 'frightened', 'happy', 'neutral', 'sad', 'surprised']
satisfaction_map = {
    'angry': 'dissatisfied',
    'disgusted': 'dissatisfied',
    'frightened': 'dissatisfied',
    'sad': 'dissatisfied',
    'happy': 'satisfied',
    'neutral': 'satisfied',
    'surprised': 'satisfied'
}

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
st.set_page_config(page_title=" ╜φ°⌂▌╫§╜φ°⌂▌╫§╜φ°⌂▌╫§╜φ°⌂▌╫§╜φ°⌂▌╫§╜φ°⌂▌╫§╜φ°⌂▌╫§╜ ")
st.title("R U Ok? || R U Happy?")
option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "Use Webcam":
    camera_input = st.camera_input("Take a picture")
    if camera_input is not None:
        image = Image.open(camera_input).convert("RGB")

# If an image is provided from either method
if image:
    st.image(image, caption="Input Image", use_container_width=True)

    model = load_model()
    input_tensor = preprocess_image(image)
    label, probabilities = predict(model, input_tensor)
    satisfaction = satisfaction_map[label]

    st.markdown(f"### Emotion: **{label}**")
    st.markdown(f"### Satisfaction Level: **{satisfaction}**")

    st.subheader("Confidence Scores")
    st.bar_chart({lbl: prob for lbl, prob in zip(class_labels, probabilities)})
