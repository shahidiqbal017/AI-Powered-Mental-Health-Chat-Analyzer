from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load sentiment and emotion models
sentiment_analyzer = pipeline("sentiment-analysis")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

@app.route('/analyze', methods=['POST'])
def analyze():
    user_message = request.json['message']

    # Analyze sentiment
    sentiment = sentiment_analyzer(user_message)[0]

    # Analyze emotion
    emotion = emotion_classifier(user_message)[0][0]

    # Generate response
    if sentiment['label'] == 'NEGATIVE' or emotion['label'] in ['sadness', 'fear', 'anger']:
        if sentiment['score'] > 0.9:
            response = "I'm sensing you're going through a tough time. Would you like to talk to a counselor?"
        else:
            response = "It's okay to feel this way. Try deep breathing. You're not alone."
    else:
        response = "I'm glad you're feeling okay! Stay positive. 😊"

    return jsonify({
        "sentiment": sentiment,
        "emotion": emotion,
        "response": response
    })

if __name__ == '__main__':
    app.run(debug=True)
