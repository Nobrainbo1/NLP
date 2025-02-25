import pandas as pd
import numpy as np

def convert_rating_to_sentiment(rating):
    """Convert 1-5 rating to sentiment (0: negative, 1: neutral, 2: positive)"""
    if rating <= 2:
        return 0  # negative
    elif rating == 3:
        return 1  # neutral
    else:
        return 2  # positive

def prepare_dataset():
    # Read the original CSV
    df = pd.read_csv('data/amazon_reviews.csv')
    
    # Create simplified dataset with only necessary columns
    simple_df = pd.DataFrame({
        'text': df['reviewText'],
        'sentiment': df['overall'].apply(convert_rating_to_sentiment)
    })
    
    # Balance the dataset
    min_count = min(simple_df['sentiment'].value_counts())
    balanced_df = pd.DataFrame()
    
    for sentiment in [0, 1, 2]:
        sentiment_samples = simple_df[simple_df['sentiment'] == sentiment].sample(n=min_count, random_state=42)
        balanced_df = pd.concat([balanced_df, sentiment_samples])
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    balanced_df.to_csv('processed_data/sentiment_dataset.csv', index=False)
    print(f"Dataset saved with {len(balanced_df)} samples ({min_count} samples per class)")

if __name__ == "__main__":
    prepare_dataset()