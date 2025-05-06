from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import torch

app = Flask(__name__)

# Sentiment model
sentiment_model = pipeline("sentiment-analysis")

# Emotion model
emotion_model = pipeline("text-classification", 
                         model="j-hartmann/emotion-english-distilroberta-base", 
                         return_all_scores=False)

# Serve the index.html page
@app.route("/")
def home():
    return render_template("index.html")

# API endpoint for analysis
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    message = data.get("message", "")

    sentiment = sentiment_model(message)[0]
    emotion = emotion_model(message)[0]

    return jsonify({
        "response": "Thank you for sharing. I'm here for you.",
        "sentiment": sentiment,
        "emotion": emotion
    })

if __name__ == "__main__":
    app.run(debug=True)
