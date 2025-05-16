import streamlit as st
import cv2
import numpy as np
from PIL import Image
from emotion_utils import detect_faces, predict_emotion, load_model
from huggingface_hub import hf_hub_download
import torch
import tempfile

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load model from Hugging Face
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = hf_hub_download(repo_id="zephyrowwa/convnxtferhehe", filename="FRconvnext_full(R)(A).pth")
model = load_model(model_path, device, weights_only=True)

st.title("test lmao")

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        faces = detect_faces(image_cv, face_cascade)
        for (x, y, w, h) in faces:
            face_crop = image_cv[y:y+h, x:x+w]
            emotion, satisfaction = predict_emotion(model, face_crop, device)
            label = f"{emotion} ({satisfaction})"
            cv2.rectangle(image_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image_cv, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        st.image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    elif uploaded_file.type.endswith("mp4"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            faces = detect_faces(frame, face_cascade)
            for (x, y, w, h) in faces:
                face_crop = frame[y:y+h, x:x+w]
                emotion, satisfaction = predict_emotion(model, face_crop, device)
                label = f"{emotion} ({satisfaction})"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
