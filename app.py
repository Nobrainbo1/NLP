from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
from model.text_process import clean_text

app = Flask(__name__)
CORS(app)

# Load the model and vectorizer
with open('processed_data/naive_bayes_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('processed_data/tfidf_features.pkl', 'rb') as f:
    data = pickle.load(f)
    vectorizer = data['vectorizer']

@app.route('/')
def home():
    return render_template('page.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        print("Received analyze request") # Debug log
        if not request.json or 'text' not in request.json:
            print("No text provided in request") # Debug log
            return jsonify({'error': 'No text provided'}), 400
            
        text = request.json['text']
        print(f"Analyzing text: {text}") # Debug log
        
        # Clean and preprocess the text
        cleaned_text = clean_text(text)
        print(f"Cleaned text: {cleaned_text}") # Debug log
        
        # Transform the text using the vectorizer
        text_vector = vectorizer.transform([cleaned_text])
        
        # Get prediction probabilities
        probabilities = model.predict_proba(text_vector)[0]
        print(f"Probabilities: {probabilities}") # Debug log
        
        # Get the predicted sentiment
        prediction = model.predict(text_vector)[0]
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        
        result = {
            'sentiment': sentiment_map[prediction],
            'probabilities': {
                'negative': float(probabilities[0]),
                'neutral': float(probabilities[1]),
                'positive': float(probabilities[2])
            }
        }
        print(f"Sending result: {result}") # Debug log
        return jsonify(result)
    
    except Exception as e:
        print(f"Error occurred: {str(e)}") # Debug log
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 