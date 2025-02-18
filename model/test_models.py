import pickle
import numpy as np
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.text_process import clean_text

# Choose which model to use by commenting/uncommenting
# MODEL_PATH = 'processed_data/naive_bayes_model.pkl'  # Naive Bayes
MODEL_PATH = 'processed_data/logistic_regression_model.pkl'  # Logistic Regression

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open('processed_data/tfidf_features.pkl', 'rb') as f:
        data = pickle.load(f)
        vectorizer = data['vectorizer']
    return model, vectorizer

def analyze_text(text, model, vectorizer):
    try:
        # Clean and preprocess the text
        cleaned_text = clean_text(text)
        
        # Transform the text using the vectorizer
        text_vector = vectorizer.transform([cleaned_text])
        
        # Get prediction probabilities
        probabilities = model.predict_proba(text_vector)[0]
        
        # Get the predicted sentiment
        prediction = model.predict(text_vector)[0]
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        
        return {
            'sentiment': sentiment_map[prediction],
            'probabilities': {
                'negative': float(probabilities[0]),
                'neutral': float(probabilities[1]),
                'positive': float(probabilities[2])
            }
        }
    
    except Exception as e:
        print(f"Error analyzing text: {str(e)}")
        return None

def main():
    try:
        model, vectorizer = load_model()
        
        while True:
            print("\nEnter text to analyze (or 'quit' to exit):")
            text = input()
            
            if text.lower() == 'quit':
                break
            
            result = analyze_text(text, model, vectorizer)
            if result:
                print(f"\nPredicted Sentiment: {result['sentiment']}")
                print("\nConfidence Scores:")
                print(f"Negative: {result['probabilities']['negative']:.2%}")
                print(f"Neutral: {result['probabilities']['neutral']:.2%}")
                print(f"Positive: {result['probabilities']['positive']:.2%}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 