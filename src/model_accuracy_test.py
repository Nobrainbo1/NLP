import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.sparse import hstack
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules from each approach
try:
    from initial_approach.text_process import clean_text
    from v2.data_processing import enhanced_clean_text
    from v3.text_processing import advanced_clean_text
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Some models may not be available for testing.")

# Test data - you can replace this with your own test dataset
TEST_DATA = [
    {'text': 'This product is terrible. I\'m very disappointed.', 'sentiment': 0},  # Negative
    {'text': 'It\'s okay. Nothing special about it.', 'sentiment': 1},  # Neutral
    {'text': 'This is amazing! I love it so much!', 'sentiment': 2},  # Positive
    {'text': 'The quality is not good, I\'m disappointed', 'sentiment': 0},  # Negative
    {'text': 'It works as expected, but nothing extraordinary', 'sentiment': 1},  # Neutral
    {'text': 'I hate this product, complete waste of money', 'sentiment': 0},  # Negative
    {'text': 'Decent product for the price', 'sentiment': 1},  # Neutral
    {'text': 'Absolutely fantastic, exceeded my expectations!', 'sentiment': 2},  # Positive
    {'text': 'Not worth the money, breaks easily', 'sentiment': 0},  # Negative
    {'text': 'Great value, highly recommend it', 'sentiment': 2}  # Positive
]

# Convert test data to DataFrame
test_df = pd.DataFrame(TEST_DATA)

# ===== Initial Approach Models =====
def load_initial_models():
    """Load models from the initial approach"""
    try:
        # Load Naive Bayes model
        with open('processed_data/naive_bayes_model.pkl', 'rb') as f:
            nb_model = pickle.load(f)
        
        # Load Logistic Regression model
        with open('processed_data/logistic_regression_model.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        
        # Load vectorizer
        with open('processed_data/tfidf_features.pkl', 'rb') as f:
            data = pickle.load(f)
            vectorizer = data['vectorizer']
        
        return {
            'naive_bayes': nb_model,
            'logistic_regression': lr_model,
            'vectorizer': vectorizer
        }
    except FileNotFoundError:
        print("Initial approach models not found.")
        return None
    except Exception as e:
        print(f"Error loading initial models: {str(e)}")
        return None

def test_initial_models(models, test_data):
    """Test initial approach models"""
    if not models:
        return {
            'naive_bayes': {'accuracy': 0, 'f1_score': 0, 'time': 0},
            'logistic_regression': {'accuracy': 0, 'f1_score': 0, 'time': 0}
        }
    
    results = {}
    
    try:
        # Prepare test data
        texts = test_data['text'].tolist()
        cleaned_texts = [clean_text(text) for text in texts]
        X_test = models['vectorizer'].transform(cleaned_texts)
        y_test = test_data['sentiment'].values
        
        # Test Naive Bayes
        start_time = time.time()
        nb_preds = models['naive_bayes'].predict(X_test)
        nb_time = time.time() - start_time
        
        nb_accuracy = accuracy_score(y_test, nb_preds)
        nb_f1 = f1_score(y_test, nb_preds, average='weighted')
        
        results['naive_bayes'] = {
            'accuracy': nb_accuracy,
            'f1_score': nb_f1,
            'time': nb_time
        }
        
        # Test Logistic Regression
        start_time = time.time()
        lr_preds = models['logistic_regression'].predict(X_test)
        lr_time = time.time() - start_time
        
        lr_accuracy = accuracy_score(y_test, lr_preds)
        lr_f1 = f1_score(y_test, lr_preds, average='weighted')
        
        results['logistic_regression'] = {
            'accuracy': lr_accuracy,
            'f1_score': lr_f1,
            'time': lr_time
        }
    except ValueError as e:
        print(f"Error testing initial models: {str(e)}")
        print("This is likely due to a feature mismatch between the model and test data.")
        results['naive_bayes'] = {'accuracy': 0, 'f1_score': 0, 'time': 0}
        results['logistic_regression'] = {'accuracy': 0, 'f1_score': 0, 'time': 0}
    except Exception as e:
        print(f"Error testing initial models: {str(e)}")
        results['naive_bayes'] = {'accuracy': 0, 'f1_score': 0, 'time': 0}
        results['logistic_regression'] = {'accuracy': 0, 'f1_score': 0, 'time': 0}
    
    return results

# ===== V2 Approach Models =====
def load_v2_models():
    """Load models from the V2 approach"""
    models = {}
    
    try:
        # Load ML models
        with open('processed_data/naive_bayes_model.pkl', 'rb') as f:
            models['naive_bayes_v2'] = pickle.load(f)
        
        with open('processed_data/logistic_regression_model.pkl', 'rb') as f:
            models['logistic_regression_v2'] = pickle.load(f)
        
        with open('processed_data/tfidf_featuresv2.pkl', 'rb') as f:
            data = pickle.load(f)
            models['vectorizer_v2'] = data['vectorizer']
    except Exception as e:
        print(f"Warning: Could not load V2 ML models: {str(e)}")
    
    try:
        # Load LSTM model
        models['lstm'] = tf.keras.models.load_model('processed_data/lstm_model.h5')
        
        with open('processed_data/tokenizer.pkl', 'rb') as f:
            models['tokenizer'] = pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load V2 LSTM model: {str(e)}")
    
    return models if models else None

def test_v2_models(models, test_data):
    """Test V2 approach models"""
    if not models:
        return {
            'naive_bayes_v2': {'accuracy': 0, 'f1_score': 0, 'time': 0},
            'logistic_regression_v2': {'accuracy': 0, 'f1_score': 0, 'time': 0},
            'lstm': {'accuracy': 0, 'f1_score': 0, 'time': 0}
        }
    
    results = {}
    texts = test_data['text'].tolist()
    y_test = test_data['sentiment'].values
    
    # Test ML models if available
    if 'vectorizer_v2' in models and 'naive_bayes_v2' in models and 'logistic_regression_v2' in models:
        try:
            cleaned_texts = [enhanced_clean_text(text) for text in texts]
            X_test = models['vectorizer_v2'].transform(cleaned_texts)
            
            # Test Naive Bayes V2
            start_time = time.time()
            nb_preds = models['naive_bayes_v2'].predict(X_test)
            nb_time = time.time() - start_time
            
            nb_accuracy = accuracy_score(y_test, nb_preds)
            nb_f1 = f1_score(y_test, nb_preds, average='weighted')
            
            results['naive_bayes_v2'] = {
                'accuracy': nb_accuracy,
                'f1_score': nb_f1,
                'time': nb_time
            }
            
            # Test Logistic Regression V2
            start_time = time.time()
            lr_preds = models['logistic_regression_v2'].predict(X_test)
            lr_time = time.time() - start_time
            
            lr_accuracy = accuracy_score(y_test, lr_preds)
            lr_f1 = f1_score(y_test, lr_preds, average='weighted')
            
            results['logistic_regression_v2'] = {
                'accuracy': lr_accuracy,
                'f1_score': lr_f1,
                'time': lr_time
            }
        except ValueError as e:
            print(f"Error testing V2 ML models: {str(e)}")
            print("This is likely due to a feature mismatch between the model and test data.")
            results['naive_bayes_v2'] = {'accuracy': 0, 'f1_score': 0, 'time': 0}
            results['logistic_regression_v2'] = {'accuracy': 0, 'f1_score': 0, 'time': 0}
        except Exception as e:
            print(f"Error testing V2 ML models: {str(e)}")
            results['naive_bayes_v2'] = {'accuracy': 0, 'f1_score': 0, 'time': 0}
            results['logistic_regression_v2'] = {'accuracy': 0, 'f1_score': 0, 'time': 0}
    else:
        results['naive_bayes_v2'] = {'accuracy': 0, 'f1_score': 0, 'time': 0}
        results['logistic_regression_v2'] = {'accuracy': 0, 'f1_score': 0, 'time': 0}
    
    # Test LSTM model if available
    if 'lstm' in models and 'tokenizer' in models:
        try:
            cleaned_texts = [enhanced_clean_text(text) for text in texts]
            sequences = models['tokenizer'].texts_to_sequences(cleaned_texts)
            X_test_padded = pad_sequences(sequences, maxlen=200, padding='post')
            
            start_time = time.time()
            lstm_preds_prob = models['lstm'].predict(X_test_padded)
            lstm_preds = np.argmax(lstm_preds_prob, axis=1)
            lstm_time = time.time() - start_time
            
            lstm_accuracy = accuracy_score(y_test, lstm_preds)
            lstm_f1 = f1_score(y_test, lstm_preds, average='weighted')
            
            results['lstm'] = {
                'accuracy': lstm_accuracy,
                'f1_score': lstm_f1,
                'time': lstm_time
            }
        except Exception as e:
            print(f"Error testing LSTM model: {str(e)}")
            results['lstm'] = {'accuracy': 0, 'f1_score': 0, 'time': 0}
    else:
        results['lstm'] = {'accuracy': 0, 'f1_score': 0, 'time': 0}
    
    return results

# ===== V3 Approach Models =====
def load_v3_models():
    """Load models from the V3 approach"""
    try:
        # Load ensemble model
        with open('processed_data/v3_ensemble_model.pkl', 'rb') as f:
            models = pickle.load(f)
        
        # Load vectorizers
        with open('processed_data/v3_vectorizers.pkl', 'rb') as f:
            vectorizers = pickle.load(f)
        
        return {
            'models': models,
            'vectorizers': vectorizers
        }
    except FileNotFoundError:
        print("V3 approach models not found.")
        return None
    except Exception as e:
        print(f"Error loading V3 models: {str(e)}")
        return None

def test_v3_models(model_data, test_data):
    """Test V3 approach models"""
    if not model_data:
        return {'ensemble_v3': {'accuracy': 0, 'f1_score': 0, 'time': 0}}
    
    try:
        models = model_data['models']
        vectorizers = model_data['vectorizers']
        
        texts = test_data['text'].tolist()
        y_test = test_data['sentiment'].values
        
        # Prepare test data
        cleaned_texts = [advanced_clean_text(text) for text in texts]
        X_word = vectorizers['word_vectorizer'].transform(cleaned_texts)
        X_char = vectorizers['char_vectorizer'].transform(cleaned_texts)
        X_combined = hstack([X_word, X_char])
        
        # Test ensemble model
        start_time = time.time()
        
        # Get predictions from both models
        ensemble_pred_proba = models['ensemble_model'].predict_proba(X_combined)
        svm_pred = models['svm_model'].predict(X_combined)
        
        # Combine predictions for each sample
        final_preds = []
        for i in range(len(texts)):
            # Convert SVM prediction to one-hot
            svm_one_hot = [0, 0, 0]
            svm_one_hot[svm_pred[i]] = 1.0
            
            # Combine predictions (weighted average)
            combined_probs = [0.7 * ensemble_pred_proba[i][j] + 0.3 * svm_one_hot[j] for j in range(3)]
            final_pred = combined_probs.index(max(combined_probs))
            final_preds.append(final_pred)
        
        ensemble_time = time.time() - start_time
        
        ensemble_accuracy = accuracy_score(y_test, final_preds)
        ensemble_f1 = f1_score(y_test, final_preds, average='weighted')
        
        return {
            'ensemble_v3': {
                'accuracy': ensemble_accuracy,
                'f1_score': ensemble_f1,
                'time': ensemble_time
            }
        }
    except ValueError as e:
        print(f"Error testing V3 ensemble model: {str(e)}")
        print("This is likely due to a feature mismatch between the model and test data.")
        return {'ensemble_v3': {'accuracy': 0, 'f1_score': 0, 'time': 0}}
    except Exception as e:
        print(f"Error testing V3 ensemble model: {str(e)}")
        return {'ensemble_v3': {'accuracy': 0, 'f1_score': 0, 'time': 0}}

# ===== Main Function =====
def main():
    print("\n===== Amazon Reviews Sentiment Analysis - Model Accuracy Test =====\n")
    
    # Load all models
    print("Loading models...")
    initial_models = load_initial_models()
    v2_models = load_v2_models()
    v3_model_data = load_v3_models()
    
    # Test all models
    print("\nTesting models on sample data...")
    initial_results = test_initial_models(initial_models, test_df)
    v2_results = test_v2_models(v2_models, test_df)
    v3_results = test_v3_models(v3_model_data, test_df)
    
    # Combine all results
    all_results = {}
    all_results.update(initial_results)
    all_results.update(v2_results)
    all_results.update(v3_results)
    
    # Print results in a table format
    print("\n===== Model Accuracy Results =====\n")
    print(f"{'Model':<25} {'Accuracy':<10} {'F1 Score':<10} {'Time (s)':<10}")
    print("-" * 55)
    
    for model_name, metrics in all_results.items():
        print(f"{model_name:<25} {metrics['accuracy']:.4f}     {metrics['f1_score']:.4f}     {metrics['time']:.6f}")
    
    # Save results to CSV for report
    results_data = []
    
    for model_name, metrics in all_results.items():
        results_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'F1 Score': metrics['f1_score'],
            'Time (s)': metrics['time']
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Create directory if it doesn't exist
    os.makedirs('benchmark_results', exist_ok=True)
    
    # Save to CSV
    results_df.to_csv('benchmark_results/model_accuracy_results.csv', index=False)
    print("\nResults saved to benchmark_results/model_accuracy_results.csv")

# Call main function when script is run directly
if __name__ == "__main__":
    main()