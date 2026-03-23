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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# 🔹 Load Models
# -------------------------------
def load_models():
    global face_model, processor, emotion_model

    if face_model is not None:
        return

    print("🔄 Loading face detection model...")
    model_path = hf_hub_download(
        repo_id="arnabdhar/YOLOv8-Face-Detection",
        filename="model.pt"
    )
    face_model = YOLO(model_path)

    print("🔄 Loading emotion model...")
    processor = AutoImageProcessor.from_pretrained(
        "dima806/face_emotions_image_detection"
    )
    emotion_model = AutoModelForImageClassification.from_pretrained(
        "dima806/face_emotions_image_detection"
    ).to(device)

    print(f"✅ Models loaded on {device}!")


# -------------------------------
# 🔹 Base64 → Image
# -------------------------------
def base64_to_image(base64_str):
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        return image
    except Exception:
        return None


# -------------------------------
# 🔹 Emotion Prediction (MULTI)
# -------------------------------
def get_emotion(face_crop):
    pil_face = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

    inputs = processor(images=pil_face, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = emotion_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]

    topk = torch.topk(probs, k=2)

    results = []
    for idx, score in zip(topk.indices, topk.values):
        if score.item() > 0.3:
            results.append({
                "emotion": emotion_model.config.id2label[idx.item()],
                "confidence": round(score.item(), 3)
            })

    if not results:
        results.append({
            "emotion": emotion_model.config.id2label[topk.indices[0].item()],
            "confidence": round(topk.values[0].item(), 3)
        })

    return results


# -------------------------------
# 🔹 Main Function
# -------------------------------
def analyze_frame_base64(base64_image):
    try:
        frame = base64_to_image(base64_image)

        if frame is None:
            return {"error": "Invalid image"}

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        output = face_model(pil_image, verbose=False)
        detections = Detections.from_ultralytics(output[0])

        if len(detections.xyxy) == 0:
            return {"error": "No face detected"}

        results = []
        h, w, _ = frame.shape

        for bbox in detections.xyxy:
            pad = 20

            x1 = max(0, int(bbox[0]) - pad)
            y1 = max(0, int(bbox[1]) - pad)
            x2 = min(w, int(bbox[2]) + pad)
            y2 = min(h, int(bbox[3]) + pad)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            emotions = get_emotion(face_crop)

            results.append({
                "emotions": emotions
            })

        if results:
            return {
                "dominant_emotion": results[0]["emotions"][0]["emotion"],
                "confidence": float(results[0]["emotions"][0]["confidence"]),
                "all_faces": results
            }

        return {"error": "Could not analyze faces"}

    except Exception as e:
        return {"error": str(e)}