from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
from scipy.sparse import hstack
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.v3.text_processing import advanced_clean_text, normalize_text_length

app = Flask(__name__)
CORS(app)

import os

# Load the V3 ensemble model and vectorizers
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'processed_data', 'v3_ensemble_model.pkl')
vectorizers_path = os.path.join(base_dir, 'processed_data', 'v3_vectorizers.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(vectorizers_path, 'rb') as f:
    vectorizers = pickle.load(f)

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
        
        # Clean and preprocess the text using V3 pipeline
        cleaned_text = advanced_clean_text(text)
        normalized_text = normalize_text_length(cleaned_text)
        print(f"Cleaned text: {normalized_text}") # Debug log
        
        # Transform using both vectorizers
        X_word = vectorizers['word_vectorizer'].transform([normalized_text])
        X_char = vectorizers['char_vectorizer'].transform([normalized_text])
        
        # Combine features
        X_combined = hstack([X_word, X_char])
        
        # Get predictions from both models
        ensemble_pred_proba = model['ensemble_model'].predict_proba(X_combined)[0]
        svm_pred = model['svm_model'].predict(X_combined)[0]
        
        # Convert SVM prediction to one-hot
        svm_one_hot = [0, 0, 0]
        svm_one_hot[svm_pred] = 1.0
        
        # Combine predictions (weighted average)
        probabilities = [0.7 * ensemble_pred_proba[i] + 0.3 * svm_one_hot[i] for i in range(3)]
        print(f"Probabilities: {probabilities}") # Debug log
        
        # Get the predicted sentiment
        prediction = np.argmax(probabilities)
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