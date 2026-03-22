import base64
import numpy as np
import cv2
import torch
from PIL import Image

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Global models (loaded once)
face_model = None
processor = None
emotion_model = None


# -------------------------------
# 🔹 Load Models (called from Flask)
# -------------------------------
def load_models():
    global face_model, processor, emotion_model

    if face_model is not None:
        return  # already loaded

    print("Loading face detection model...")
    model_path = hf_hub_download(
        repo_id="arnabdhar/YOLOv8-Face-Detection",
        filename="model.pt"
    )
    face_model = YOLO(model_path)

    print("Loading emotion model...")
    processor = AutoImageProcessor.from_pretrained(
        "dima806/face_emotions_image_detection"
    )
    emotion_model = AutoModelForImageClassification.from_pretrained(
        "dima806/face_emotions_image_detection"
    )

    print("✅ Models loaded successfully!")


# -------------------------------
# 🔹 Convert Base64 → OpenCV Image
# -------------------------------
def base64_to_image(base64_str):
    try:
        # Remove metadata like "data:image/jpeg;base64,..."
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        return image
    except Exception as e:
        return None


# -------------------------------
# 🔹 Emotion Prediction
# -------------------------------
def get_emotion(face_crop):
    pil_face = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

    inputs = processor(images=pil_face, return_tensors="pt")

    with torch.no_grad():
        outputs = emotion_model(**inputs)

    predicted_class = outputs.logits.argmax(-1).item()
    emotion = emotion_model.config.id2label[predicted_class]

    probs = torch.softmax(outputs.logits, dim=-1)
    confidence = probs[0][predicted_class].item()

    return emotion, confidence


# -------------------------------
# 🔹 Main Function (Used by Flask)
# -------------------------------
def analyze_frame_base64(base64_image):
    try:
        frame = base64_to_image(base64_image)

        if frame is None:
            return {"error": "Invalid image"}

        # Convert to PIL for YOLO
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Face detection
        output = face_model(pil_image, verbose=False)
        detections = Detections.from_ultralytics(output[0])

        if len(detections.xyxy) == 0:
            return {
                "error": "No face detected"
            }

        results = []

        for bbox in detections.xyxy:
            x1, y1, x2, y2 = map(int, bbox)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            emotion, confidence = get_emotion(face_crop)

            results.append({
                "emotion": emotion,
                "confidence": round(confidence, 3)
            })

        # Return dominant (first face)
        if results:
            return {
                "dominant_emotion": results[0]["emotion"],
"confidence": float(results[0].get("confidence", 0)),
                "all_faces": results
            }

        return {"error": "Could not analyze faces"}

    except Exception as e:
        return {"error": str(e)}