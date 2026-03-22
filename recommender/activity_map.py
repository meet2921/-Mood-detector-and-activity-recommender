import random

activity_map = {
    "happy": [
        "Start your hardest task",
        "Work on your main goal for 25 minutes",
        "Plan your day clearly"
    ],
    "sad": [
        "Listen to calming music",
        "Take a short walk outside",
        "Call a close friend"
    ],
    "angry": [
        "Take 5 deep breaths",
        "Do light stretching",
        "Write down what is bothering you"
    ],
    "fear": [
        "Break your task into small steps",
        "Do grounding breathing exercise",
        "Talk to someone you trust"
    ],
    "surprise": [
        "Take a moment to reflect",
        "Write down your thoughts",
        "Stay calm and think clearly"
    ],
    "disgust": [
        "Clean your workspace",
        "Drink water and refresh yourself",
        "Take a small break"
    ],
    "neutral": [
        "Do a small productive task",
        "Organize your desk",
        "Read something useful"
    ]
}


def get_activities_for_emotion(emotion):
    emotion = emotion.lower()

    if emotion not in activity_map:
        return {
            "response": "Take a mindful pause.",
            "activities": ["Relax for a moment"],
            "message": "Couldn't detect emotion clearly."
        }

    activities = activity_map[emotion]

    return {
        "response": random.choice(activities),
        "activities": activities,
        "message": f"You seem {emotion}. Here are some suggestions."
    }