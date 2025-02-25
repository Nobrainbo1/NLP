import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_processing import enhanced_clean_text  # Import our text processing function

def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        # Load the model
        model = tf.keras.models.load_model('processed_data/lstm_model.h5')
        
        # Load the tokenizer
        with open('processed_data/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
            
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {str(e)}")
        return None, None

def predict_sentiment(text, model, tokenizer):
    """Predict sentiment for a given text"""
    # Clean the text
    cleaned_text = enhanced_clean_text(text)
    
    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    
    # Pad sequence
    padded = pad_sequences(sequence, maxlen=200, padding='post')
    
    # Get prediction
    prediction = model.predict(padded, verbose=0)[0]
    
    # Get sentiment label and confidence
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    predicted_sentiment = sentiment_labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return predicted_sentiment, prediction, confidence

def print_sentiment_analysis(text, prediction, confidence):
    """Print detailed sentiment analysis"""
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    print("\n" + "="*50)
    print(f"Text: '{text}'")
    print("-"*50)
    print("Sentiment Analysis:")
    print(f"Predicted Sentiment: {sentiment_labels[np.argmax(prediction)]}")
    print(f"Confidence: {confidence:.2%}")
    print("\nProbabilities:")
    for label, prob in zip(sentiment_labels, prediction):
        print(f"{label}: {prob:.2%}")
    print("="*50)

def main():
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer. Please make sure they exist in the processed_data directory.")
        return
    
    print("\nSentiment Analysis Tool")
    print("Enter 'quit' or 'exit' to end the program")
    print("="*50)
    
    while True:
        # Get user input
        text = input("\nEnter text to analyze: ").strip()
        
        # Check for exit command
        if text.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break
        
        # Skip empty input
        if not text:
            print("Please enter some text to analyze.")
            continue
        
        try:
            # Get prediction
            predicted_sentiment, prediction, confidence = predict_sentiment(text, model, tokenizer)
            
            # Print analysis
            print_sentiment_analysis(text, prediction, confidence)
            
        except Exception as e:
            print(f"Error analyzing text: {str(e)}")
            print("Please try again with different text.")

if __name__ == "__main__":
    main() 