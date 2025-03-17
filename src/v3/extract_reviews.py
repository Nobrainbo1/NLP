import pandas as pd
import json
import os
import numpy as np
from sklearn.utils import resample

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def convert_to_sentiment(rating):
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

def extract_reviews(file_path, text_column='reviewText', rating_column='overall'):
    """Generic function to extract reviews from CSV files with configurable column names.
    
    Args:
        file_path (str): Path to the CSV file
        text_column (str): Name of the column containing review text
        rating_column (str): Name of the column containing ratings
        
    Returns:
        pd.DataFrame: DataFrame containing review_text, rating, and sentiment
    """
    try:
        # Try different encodings if utf-8 fails
        encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
        
        # Try each encoding until one works
        for encoding in encodings:
            try:
                # Read CSV file with the current encoding
                df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='warn')
                break  # If successful, break the loop
            except Exception as e:
                if encoding == encodings[-1]:  # If this is the last encoding to try
                    raise e  # Re-raise the exception if all encodings fail
                continue  # Otherwise try the next encoding
        
        # Verify required columns exist
        if text_column not in df.columns or rating_column not in df.columns:
            raise ValueError(f"Required columns {text_column} and/or {rating_column} not found in {file_path}")
        
        # Extract reviews and ratings
        reviews_df = pd.DataFrame({
            'review_text': df[text_column],
            'rating': df[rating_column]
        }).dropna()
        
        # Basic text cleaning to standardize before deduplication
        reviews_df['review_text'] = reviews_df['review_text'].astype(str).apply(lambda x: x.strip())
        
        # Convert ratings to sentiment
        reviews_df['sentiment'] = reviews_df['rating'].apply(convert_to_sentiment)
        return reviews_df
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return pd.DataFrame()

def extract_reviews_from_amazon():
    """Extract reviews from amazon_reviews.csv file.
    
    Returns:
        pd.DataFrame: DataFrame containing review_text, rating, and sentiment
    """
    return extract_reviews('data\\amazon_reviews.csv')

def balance_dataset(df):
    # Separate by sentiment
    negative = df[df['sentiment'] == 0]
    neutral = df[df['sentiment'] == 1]
    positive = df[df['sentiment'] == 2]
    
    # Find the minimum count
    min_count = min(len(negative), len(neutral), len(positive))
    
    # Down-sample each class to min_count
    negative_balanced = resample(negative, n_samples=min_count, random_state=42)
    neutral_balanced = resample(neutral, n_samples=min_count, random_state=42)
    positive_balanced = resample(positive, n_samples=min_count, random_state=42)
    
    # Combine balanced datasets
    return pd.concat([negative_balanced, neutral_balanced, positive_balanced])

def main():
    ensure_directory_exists('v3')
    ensure_directory_exists('processed_data')
    
    try:
        # Extract reviews from amazon_reviews.csv only
        df_reviews = extract_reviews_from_amazon()
        
        # Remove any empty reviews
        initial_count = len(df_reviews)
        df_reviews = df_reviews[df_reviews['review_text'].str.len() > 0]
        empty_removed = initial_count - len(df_reviews)
        
        # Standardize text for better duplicate detection
        df_reviews['review_text_normalized'] = df_reviews['review_text'].str.lower().str.strip()
        
        # Drop duplicates with more detailed tracking
        before_dedup = len(df_reviews)
        df_reviews = df_reviews.drop_duplicates(subset=['review_text_normalized'])
        after_dedup = len(df_reviews)
        duplicates_removed = before_dedup - after_dedup
        
        # Remove the temporary normalization column
        df_reviews = df_reviews.drop(columns=['review_text_normalized'])
        
        # Balance the dataset
        print("\nPre-balancing Sentiment Distribution:")
        print(df_reviews['sentiment'].value_counts().sort_index())
        
        df_balanced = balance_dataset(df_reviews)
        
        # Shuffle the final dataset
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save to CSV
        output_path = 'data/balanced_reviews_v2.csv'
        df_balanced.to_csv(output_path, index=False, encoding='utf-8')
        
        # Print statistics
        print("\nDataset Processing Statistics:")
        print(f"Initial reviews: {initial_count}")
        print(f"Empty reviews removed: {empty_removed}")
        print(f"Duplicate reviews removed: {duplicates_removed}")
        print(f"Reviews after deduplication: {after_dedup}")
        print(f"Final balanced reviews: {len(df_balanced)}")
        
        print("\nFinal Sentiment Distribution:")
        print(df_balanced['sentiment'].value_counts().sort_index())
        print(f"\nData saved to {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()