import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"
import keras
import tensorflow as tf
from deepface import DeepFace
import cv2


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    emotion = result[0]['dominant_emotion']

    print("Detected Emotion:", emotion)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()