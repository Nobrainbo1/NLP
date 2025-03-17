import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns

def create_ensemble_model(X_word, X_char, y, test_size=0.2, random_state=42):
    """
    Create and train an ensemble model combining multiple classifiers
    
    Args:
        X_word: Word-based TF-IDF features
        X_char: Character-based TF-IDF features
        y: Target labels
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing the trained model, test data, and evaluation metrics
    """
    # Combine word and character features
    X_combined = hstack([X_word, X_char])
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create individual classifiers
    nb_classifier = MultinomialNB(alpha=0.1)  # Original alpha value
    class_weights = {0: 3.0, 1: 2.0, 2: 0.1}  # Original class weights
    lr_classifier = LogisticRegression(C=10.0, max_iter=1000, class_weight=class_weights)
    svm_classifier = LinearSVC(C=1.0, class_weight=class_weights, dual=False, max_iter=10000)
    
    # Create ensemble model with soft voting (probability-based)
    # For LinearSVC, we need to use a different approach since it doesn't have predict_proba
    estimators = [
        ('naive_bayes', nb_classifier),
        ('logistic_regression', lr_classifier)
    ]
    
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    
    # Train the ensemble model
    ensemble.fit(X_train, y_train)
    
    # Train SVM separately (since it doesn't have predict_proba)
    svm_classifier.fit(X_train, y_train)
    
    # Evaluate the models
    ensemble_pred = ensemble.predict(X_test)
    svm_pred = svm_classifier.predict(X_test)
    
    # Combine predictions (simple averaging of predictions)
    # Convert predictions to one-hot encoding
    ensemble_one_hot = np.eye(3)[ensemble_pred]
    svm_one_hot = np.eye(3)[svm_pred]
    
    # Average the predictions (giving more weight to ensemble)
    combined_probs = 0.7 * ensemble.predict_proba(X_test) + 0.3 * svm_one_hot
    final_pred = np.argmax(combined_probs, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, final_pred)
    class_report = classification_report(y_test, final_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, final_pred)
    
    # Create result dictionary
    result = {
        'ensemble_model': ensemble,
        'svm_model': svm_classifier,
        'X_test': X_test,
        'y_test': y_test,
        'predictions': final_pred,
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }
    
    return result

def visualize_results(result, output_dir='progress_report'):
    """
    Visualize the model results
    
    Args:
        result: Dictionary containing model evaluation results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        result['confusion_matrix'], 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Negative', 'Neutral', 'Positive'],
        yticklabels=['Negative', 'Neutral', 'Positive']
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/v3_confusion_matrix.png')
    plt.close()
    
    # Plot class-wise metrics
    metrics = ['precision', 'recall', 'f1-score']
    class_names = ['Negative', 'Neutral', 'Positive']
    
    plt.figure(figsize=(10, 6))
    for i, cls in enumerate(['0', '1', '2']):
        values = [result['classification_report'][cls][m] for m in metrics]
        plt.bar(
            [f"{m} ({class_names[i]})" for m in metrics],
            values,
            alpha=0.7,
            label=class_names[i]
        )
    
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.ylim(0, 1.0)
    plt.ylabel('Score')
    plt.title('Class-wise Performance Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/v3_class_metrics.png')
    plt.close()

def predict_sentiment(text, models_data, vectorizers):
    """
    Predict sentiment for new text
    
    Args:
        text: Text to predict sentiment for
        models_data: Dictionary containing trained models
        vectorizers: Dictionary containing word and character vectorizers
        
    Returns:
        dict: Dictionary containing prediction results
    """
    # Clean and vectorize the text
    from text_processing import advanced_clean_text, SENTIMENT_WORDS
    
    cleaned_text = advanced_clean_text(text)
    
    # Special handling for short texts with strong sentiment words
    words = text.lower().split()
    is_short_text = len(words) <= 5  # Original threshold for short text detection
    
    # Check for strong negative sentiment words in short texts
    strong_negative_words = {'terrible', 'awful', 'horrible', 'hate', 'worst', 'bad', 'poor', 'disappointing', 'disappointed', 
                           'useless', 'waste', 'problem', 'defective', 'broken', 'fail', 'failed', 'sucks', 'suck',
                           'difficult', 'impossible', 'error', 'faulty', 'cheap', 'expensive', 'overpriced', 'avoid', 'return',
                           'not good', 'not worth', 'not recommend', 'not happy', 'not satisfied', 'not working', 'not useful'}
    
    # Check for individual negative words
    has_strong_negative = any(word in strong_negative_words for word in words)
    
    # Also check for negative phrases in the original text
    negative_phrases = ['waste of money', 'not good', 'not worth', 'not recommend', 'not happy', 'not satisfied', 'not working']
    has_negative_phrase = any(phrase in text.lower() for phrase in negative_phrases)
    
    # Transform using both vectorizers
    X_word = vectorizers['word_vectorizer'].transform([cleaned_text])
    X_char = vectorizers['char_vectorizer'].transform([cleaned_text])
    
    # Combine features
    X_combined = hstack([X_word, X_char])
    
    # Get predictions from both models
    ensemble_pred_proba = models_data['ensemble_model'].predict_proba(X_combined)[0]
    svm_pred = models_data['svm_model'].predict(X_combined)[0]
    
    # Convert SVM prediction to one-hot
    svm_one_hot = np.zeros(3)
    svm_one_hot[svm_pred] = 1.0
    
    # Combine predictions (weighted average)
    combined_probs = 0.7 * ensemble_pred_proba + 0.3 * svm_one_hot
    
    # Boost negative sentiment for short texts with strong negative words or phrases
    if (is_short_text and (has_strong_negative or has_negative_phrase)) or has_negative_phrase:
        # Increase negative probability and decrease others proportionally
        boost_factor = 5.0  # Original boost factor
        combined_probs[0] *= boost_factor
        # Normalize probabilities to sum to 1
        combined_probs = combined_probs / np.sum(combined_probs)
    
    final_pred = np.argmax(combined_probs)
    
    # Map prediction to sentiment label
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    predicted_sentiment = sentiment_labels[final_pred]
    
    # Calculate confidence scores (as percentages)
    confidence_scores = combined_probs * 100
    
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

def main():
    try:
        # Load TF-IDF features
        with open('processed_data/v3_tfidf_features.pkl', 'rb') as f:
            features = pickle.load(f)
        
        # Create and train ensemble model
        result = create_ensemble_model(
            features['X_word'], 
            features['X_char'], 
            features['y']
        )
        
        # Save the trained model
        model_data = {
            'ensemble_model': result['ensemble_model'],
            'svm_model': result['svm_model']
        }
        
        with open('processed_data/v3_ensemble_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save vectorizers separately for easier loading
        vectorizers = {
            'word_vectorizer': features['word_vectorizer'],
            'char_vectorizer': features['char_vectorizer']
        }
        
        with open('processed_data/v3_vectorizers.pkl', 'wb') as f:
            pickle.dump(vectorizers, f)
        
        # Visualize results
        visualize_results(result)
        
        # Print evaluation metrics
        print("\nEnsemble Model Evaluation:")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print("\nClassification Report:")
        for cls, metrics in result['classification_report'].items():
            if cls in ['0', '1', '2']:
                sentiment = ['Negative', 'Neutral', 'Positive'][int(cls)]
                print(f"{sentiment}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-score: {metrics['f1-score']:.4f}")
        
        # Test with sample reviews
        print("\nSample Predictions:")
        test_reviews = [
            "This product is terrible",
            "It's okay, nothing special",
            "Amazing product, I love it!"
        ]
        
        for review in test_reviews:
            prediction = predict_sentiment(review, model_data, vectorizers)
            print(f"\nReview: '{review}'")
            print(f"Predicted sentiment: {prediction['sentiment']} (confidence: {prediction['confidence'][prediction['prediction']]:.0f}%)")
            print("Probabilities:")
            print(f"  Negative: {prediction['confidence_negative']:.2f}%")
            print(f"  Neutral:  {prediction['confidence_neutral']:.2f}%")
            print(f"  Positive: {prediction['confidence_positive']:.2f}%")
        
        print("\nModel training and evaluation completed!")
        print("Models saved to processed_data/v3_ensemble_model.pkl and v3_vectorizers.pkl")
        
    except FileNotFoundError:
        print("Error: Could not find input files. Make sure to run text_processing.py and feature_engineering.py first.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()