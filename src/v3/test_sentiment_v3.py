import sys
import os
import pickle
from scipy.sparse import hstack

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the v3 module
from v3.text_processing import advanced_clean_text

def load_models(model_path='processed_data/v3_ensemble_model.pkl', 
               vectorizers_path='processed_data/v3_vectorizers.pkl'):
    """Load the trained models and vectorizers"""
    try:
        # Load model and vectorizers
        with open(model_path, 'rb') as f:
            models = pickle.load(f)
        
        with open(vectorizers_path, 'rb') as f:
            vectorizers = pickle.load(f)
            
        return models, vectorizers
    except FileNotFoundError:
        print(f"Error: Model files not found. Make sure to train the model first.")
        return None, None
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None

def predict_sentiment(text, models, vectorizers):
    """Predict sentiment for a given text"""
    # Clean the text
    cleaned_text = advanced_clean_text(text)
    
    # Transform using both vectorizers
    X_word = vectorizers['word_vectorizer'].transform([cleaned_text])
    X_char = vectorizers['char_vectorizer'].transform([cleaned_text])
    
    # Combine features
    X_combined = hstack([X_word, X_char])
    
    # Get predictions from both models
    ensemble_pred_proba = models['ensemble_model'].predict_proba(X_combined)[0]
    svm_pred = models['svm_model'].predict(X_combined)[0]
    
    # Convert SVM prediction to one-hot
    svm_one_hot = [0, 0, 0]
    svm_one_hot[svm_pred] = 1.0
    
    # Combine predictions (weighted average)
    combined_probs = [0.7 * ensemble_pred_proba[i] + 0.3 * svm_one_hot[i] for i in range(3)]
    final_pred = combined_probs.index(max(combined_probs))
    
    # Map prediction to sentiment label
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    predicted_sentiment = sentiment_labels[final_pred]
    
    # Calculate confidence scores (as percentages)
    confidence_scores = [prob * 100 for prob in combined_probs]
    
    return {
        'text': text,
        'cleaned_text': cleaned_text,
        'prediction': final_pred,
        'sentiment': predicted_sentiment,
        'confidence': confidence_scores,
        'confidence_negative': confidence_scores[0],
        'confidence_neutral': confidence_scores[1],
        'confidence_positive': confidence_scores[2]
    }

def print_sentiment_analysis(result):
    """Print detailed sentiment analysis"""
    print(f"\nText: '{result['text']}'")
    print(f"Cleaned text: '{result['cleaned_text']}'")
    print(f"\nPredicted sentiment: {result['sentiment']} (confidence: {result['confidence'][result['prediction']]:.0f}%)")
    print("Confidence scores:")
    print(f"  Negative: {result['confidence_negative']:.2f}%")
    print(f"  Neutral:  {result['confidence_neutral']:.2f}%")
    print(f"  Positive: {result['confidence_positive']:.2f}%")

def main():
    print("\nV3 Sentiment Analysis Tool")
    print("==========================")
    
    # Load models
    models, vectorizers = load_models()
    if not models or not vectorizers:
        return
    
    # Test with sample reviews
    sample_reviews = [
        "This product is terrible",
        "It's okay, nothing special",
        "Amazing product, I love it!",
        "The quality is not good, I'm disappointed",
        "It works as expected, but nothing extraordinary",
        "I hate this product, complete waste of money"
    ]
    
    print("\nAnalyzing sample reviews...\n")
    for review in sample_reviews:
        result = predict_sentiment(review, models, vectorizers)
        print_sentiment_analysis(result)
        print("-" * 50)
    
    # Interactive mode
    print("\nEnter your own text for sentiment analysis (or 'q' to quit):")
    while True:
        text = input("\nEnter text: ")
        if text.lower() == 'q':
            break
        
        result = predict_sentiment(text, models, vectorizers)
        print_sentiment_analysis(result)

if __name__ == "__main__":
    main()