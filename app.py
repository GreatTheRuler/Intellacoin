from flask import Flask, request, jsonify
from IntellaCoinAI import IntellaCoinAI

app = Flask(__name__)
ai = IntellaCoinAI()

@app.route('/predict_price', methods=['POST'])
def predict_price():
    data = request.json['data']
    prediction = ai.predict_price_arima(data)
    return jsonify({'prediction': prediction})

@app.route('/detect_fraud', methods=['POST'])
def detect_fraud():
    transaction_data = request.json['transaction_data']
    is_safe = ai.detect_fraud(transaction_data)
    return jsonify({'is_safe': is_safe})

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    text = request.json['text']
    sentiment = ai.analyze_sentiment(text)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
