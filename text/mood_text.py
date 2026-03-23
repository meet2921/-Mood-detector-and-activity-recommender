from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ctransformers import AutoModelForCausalLM
import torch
import random

# =========================
# 🔥 LOAD EMOTION MODEL
# =========================

print("🔄 Loading emotion model...")
emotion_tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
emotion_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
print("✅ Emotion model loaded!")

# =========================
# 🔥 LOAD MISTRAL (LOCAL)
# =========================

print("🔄 Loading Mistral model...")

llm = AutoModelForCausalLM.from_pretrained(
    "models/mistral.gguf",   # 👈 your path
    model_type="mistral",
    gpu_layers=0  # set >0 if you have GPU
)

print("✅ Mistral loaded!")

# =========================
# 🎯 LABELS
# =========================

labels = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral"
]

# =========================
# 🧠 EMOTION DETECTION
# =========================

def detect_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True)
    outputs = emotion_model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0]

    emotions = []
    for i, p in enumerate(probs):
        if p > 0.4:
            emotions.append(labels[i])

    return emotions if emotions else ["neutral"]

# =========================
# 🤖 MISTRAL RESPONSE
# =========================

def generate_response(user_input, emotions):
    emotion_text = ", ".join(emotions)

    prompt = f"""
<s>[INST]
You are a friendly and supportive chatbot.

User emotion: {emotion_text}
User message: "{user_input}"

Respond naturally like a human in 1-2 lines.
Do NOT give instructions.
[/INST]
"""

    response = llm(
        prompt,
        max_new_tokens=120,
        temperature=0.8,
        top_p=0.9
    )

    return clean_response(response)

# =========================
# 🧹 CLEAN RESPONSE
# =========================

def clean_response(text):
    bad_phrases = [
        "instruction", "user:", "assistant:", "[inst]"
    ]

    text_lower = text.lower()

    for phrase in bad_phrases:
        if phrase in text_lower:
            return fallback_response()

    return text.strip()

# =========================
# 🔁 FALLBACK
# =========================

def fallback_response():
    return random.choice([
        "I'm here for you. Want to talk about it?",
        "That sounds interesting. Tell me more 😊",
        "I understand. How are you feeling exactly?",
        "Hmm, what’s going on in your mind?"
    ])

# =========================
# 🎯 MAIN FUNCTION
# =========================

def analyze_mood_text(text):
    text_lower = text.lower().strip()

    # Greeting
    if text_lower in ["hi", "hello", "hey"]:
        return {
            "emotion": ["neutral"],
            "response": random.choice([
                "Hey! 😊 How are you feeling today?",
                "Hello! What's on your mind?",
                "Hi there! Tell me how you're feeling."
            ]),
            "activities": [],
            "message": ""
        }

    if not text_lower:
        return {
            "emotion": ["neutral"],
            "response": "Tell me how you're feeling 😊",
            "activities": [],
            "message": ""
        }

    emotions = detect_emotion(text)
    response = generate_response(text, emotions)

    return {
        "emotion": emotions,
        "response": response,
        "activities": [],
        "message": ""
    }