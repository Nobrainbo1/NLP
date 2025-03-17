import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

def create_advanced_tfidf_features(df, text_column='cleaned_text', output_path=None):
    """
    Create advanced TF-IDF features with better handling of sentiment-critical words
    
    Args:
        df: DataFrame containing the text data
        text_column: Column name containing the cleaned text
        output_path: Path to save the features and vectorizer
        
    Returns:
        dict: Dictionary containing features X, labels y, and the vectorizer
    """
    # Create advanced TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=50000,      # Increased from previous versions
        min_df=3,                # Reduced to capture more rare but important terms
        max_df=0.95,             # Remove very common words
        ngram_range=(1, 3),      # Include unigrams, bigrams, and trigrams
        analyzer='word',         # Use word n-grams
        sublinear_tf=True,       # Apply sublinear scaling (log scaling)
        use_idf=True,            # Use inverse document frequency
        smooth_idf=True,         # Smooth IDF weights
        strip_accents='unicode', # Remove accents
        token_pattern=r'\b\w+\b', # Match word boundaries better
    )
    
    # Create character n-grams vectorizer for handling typos and variations
    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb',      # Character n-grams within word boundaries
        ngram_range=(2, 5),      # Character n-grams from 2 to 5 characters
        max_features=10000,      # Limit features
        min_df=3,                # Minimum document frequency
        sublinear_tf=True        # Apply sublinear scaling
    )
    
    # Fit and transform the text data
    X_word = vectorizer.fit_transform(df[text_column])
    X_char = char_vectorizer.fit_transform(df[text_column])
    
    # Get sentiment labels
    y = df['sentiment'].values
    
    # Create feature dictionary
    features = {
        'X_word': X_word,
        'X_char': X_char, 
        'y': y,
        'word_vectorizer': vectorizer,
        'char_vectorizer': char_vectorizer
    }
    
    # Save features if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"Features saved to {output_path}")
    
    return features

def analyze_feature_importance(vectorizer, feature_type="word"):
    """
    Analyze and print the most important features from the vectorizer
    
    Args:
        vectorizer: Fitted TfidfVectorizer
        feature_type: Type of features (word or char)
    """
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate the average TF-IDF score for each feature
    tfidf_mean = np.array(vectorizer.idf_)
    
    # Sort features by importance (inverse of IDF - lower IDF means more common)
    indices = np.argsort(tfidf_mean)
    
    # Print most common features (lowest IDF)
    print(f"\nMost common {feature_type} features:")
    for i in indices[:20]:
        print(f"{feature_names[i]}: {tfidf_mean[i]:.4f}")
    
    # Print most distinctive features (highest IDF)
    print(f"\nMost distinctive {feature_type} features:")
    for i in indices[-20:]:
        print(f"{feature_names[i]}: {tfidf_mean[i]:.4f}")

def main():
    try:
        # Load the cleaned reviews
        reviews = pd.read_csv("processed_data/v3_cleaned_reviews.csv")
        
        # Create advanced TF-IDF features
        features = create_advanced_tfidf_features(
            reviews, 
            text_column='cleaned_text',
            output_path='processed_data/v3_tfidf_features.pkl'
        )
        
        # Analyze feature importance
        analyze_feature_importance(features['word_vectorizer'], "word")
        analyze_feature_importance(features['char_vectorizer'], "character")
        
        print("\nFeature engineering completed!")
        print(f"Word features shape: {features['X_word'].shape}")
        print(f"Character features shape: {features['X_char'].shape}")
        
    except FileNotFoundError:
        print("Error: Could not find input file. Make sure v3_cleaned_reviews.csv is in the processed_data folder.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()