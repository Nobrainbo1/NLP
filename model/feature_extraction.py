import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def main():
    try:
        # Load the cleaned reviews
        reviews = pd.read_csv("processed_data/cleaned_reviews.csv")
        
        """newly added"""
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=100000,  # Increase features
            min_df=5,           # Remove rare words
            max_df=0.95,        # Remove very common words
            ngram_range=(1, 2)  # Include bigrams
        )
        X = vectorizer.fit_transform(reviews['cleaned_text'])
        y = reviews['sentiment']
        
        # Save the features and vectorizer
        with open('processed_data/tfidf_features.pkl', 'wb') as f:
            pickle.dump({'X': X, 'y': y, 'vectorizer': vectorizer}, f)
        print("Feature extraction completed!")
        
    except FileNotFoundError:
        print("Error: Could not find input file. Make sure cleaned_reviews.csv is in the processed_data folder.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 