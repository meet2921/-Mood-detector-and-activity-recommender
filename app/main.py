from flask import Flask, render_template, request, jsonify
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion.test_emotion import analyze_frame_base64, load_models
from recommender.activity_map import get_activities_for_emotion
from text.mood_text import analyze_mood_text

app = Flask(__name__,
    template_folder='../templates',
    static_folder='../static'
)

print("Loading models...")
load_models()
print("Ready!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze-face', methods=['POST'])
def analyze_face():
    try:
        data = request.json
        result = analyze_frame_base64(data.get('image'))
        if 'error' not in result:
            recs = get_activities_for_emotion(result['dominant_emotion'])
            result['response'] = recs['response']
            result['activities'] = recs['activities']
            result['message'] = recs['message']
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)})
    

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    try:
        data = request.json
        user_input = data.get('text', '')

        # 🔒 HARD FILTER (this is the fix)
        mood_keywords = [
            "sad","happy","bored","tired","lonely",
            "stress","anxious","scrolling","reels",
            "nothing to do","feeling low"
        ]

        if not any(word in user_input.lower() for word in mood_keywords):
            return jsonify({
                "emotion": None,
                "response": "I only help with mood and activity suggestions.",
                "activities": [],
                "message": "Tell me how you're feeling 😊"
            })

        # ✅ If valid → continue your existing logic
        result = analyze_mood_text(user_input)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=False, port=5000)