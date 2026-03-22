# mood_text.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ✅ Load model once (IMPORTANT)
print("🔄 Loading FLAN-T5 model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
print("✅ Model loaded!")


# 🤖 FLAN response
def ask_flan(user_input: str) -> str:
    try:
        prompt = f"""
You are a mood activity assistant.

User: {user_input}

RULES:
- If mood-related → suggest exactly 3 short activities
- Format ONLY like:
• Activity 1
• Activity 2
• Activity 3
- No explanation
- If not mood-related → say: I only help with mood and activity suggestions.
"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = model.generate(
            **inputs,
            max_length=100,
            temperature=0.7,
            do_sample=True
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("✅ FLAN RAW:", response)  # DEBUG

        return clean_response(response)

    except Exception as e:
        print("❌ FLAN Error:", e)
        return default_response()


# 🧹 Clean output (VERY IMPORTANT)
def clean_response(text: str) -> str:
    import re

    # Try bullet points already in response
    bullets = re.findall(r'[•\-\*]\s*(.+)', text)
    if len(bullets) >= 3:
        return f"• {bullets[0].strip()}\n• {bullets[1].strip()}\n• {bullets[2].strip()}"

    # Try comma or semicolon separated (FLAN often outputs these)
    parts = re.split(r'[,;]', text)
    parts = [p.strip().strip('.') for p in parts if p.strip()]
    if len(parts) >= 3:
        return f"• {parts[0]}\n• {parts[1]}\n• {parts[2]}"

    # Try newline split as last resort
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) >= 3:
        return f"• {lines[0]}\n• {lines[1]}\n• {lines[2]}"

    return default_response()

# 🔁 fallback (safe)
def default_response():
    return "• Go for a walk\n• Listen to music\n• Take a break"


# 🎯 Main function used by Flask
def analyze_mood_text(text):
    text_lower = text.lower().strip()

    if not text_lower:
        return {
            "emotion": "neutral",
            "response": "Tell me how you're feeling 😊",
            "activities": [],
            "message": ""
        }

    # 🚫 HARD FILTER (blocks coding etc.)
    blocked = ["code", "python", "java", "react", "math", "program"]
    if any(word in text_lower for word in blocked):
        return {
            "emotion": None,
            "response": "I only help with mood and activity suggestions.",
            "activities": [],
            "message": ""
        }

    # ⚡ FAST RULE-BASED (improves quality a lot)
    if any(word in text_lower for word in ["sad", "cry", "depressed", "upset"]):
        return {
            "emotion": "sad",
            "response": "• Talk to a friend\n• Listen to calm music\n• Go for a short walk",
            "activities": [],
            "message": ""
        }

    if any(word in text_lower for word in ["happy", "excited"]):
        return {
            "emotion": "happy",
            "response": "• Share your happiness\n• Capture the moment\n• Do something creative",
            "activities": [],
            "message": ""
        }

    if any(word in text_lower for word in ["angry", "frustrated"]):
        return {
            "emotion": "angry",
            "response": "• Take deep breaths\n• Go for a walk\n• Write your thoughts",
            "activities": [],
            "message": ""
        }

    # 🤖 FLAN fallback (AI)
    ai_response = ask_flan(text)

    return {
        "emotion": "ai",
        "response": ai_response,
        "activities": [],
        "message": ""
    }