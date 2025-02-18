import pandas as pd
import numpy as np

def convert_to_sentiment(rating):
    """
    Convert 5-star rating to sentiment class
    1-2 stars = Negative (0)
    3 stars = Neutral (1)
    4-5 stars = Positive (2)
    """
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

def create_balanced_dataset(reviews, samples_per_class=100):
    """
    Create a balanced dataset by sampling equal numbers from each sentiment class
    """
    # First, separate reviews by sentiment
    neg_reviews = reviews[reviews['sentiment'] == 0]
    neu_reviews = reviews[reviews['sentiment'] == 1]
    pos_reviews = reviews[reviews['sentiment'] == 2]
    
    # Calculate the minimum number of samples available
    min_samples = min(samples_per_class,
                        len(neg_reviews),
                        len(neu_reviews),
                        len(pos_reviews))
    
    print(f"\nSampling {min_samples} reviews from each sentiment class...")
    
    # Sample equally from each class
    neg_sample = neg_reviews.sample(n=min_samples, random_state=42)
    neu_sample = neu_reviews.sample(n=min_samples, random_state=42)
    pos_sample = pos_reviews.sample(n=min_samples, random_state=42)
    
    # Combine the samples
    balanced_df = pd.concat([neg_sample, neu_sample, pos_sample])
    
    # Shuffle the combined dataset
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

def main():
    # Load the data
    print("Loading the dataset...")
    reviews = pd.read_csv("amazon_reviews.csv", encoding='utf-8')
    
    # Add new sentiment column
    print("Converting ratings to sentiment...")
    reviews['sentiment'] = reviews['overall'].apply(convert_to_sentiment)
    
    # Display original distribution
    sentiment_counts = reviews['sentiment'].value_counts().sort_index()
    print("\nOriginal Sentiment Distribution:")
    print("-" * 30)
    print("Negative (0):", sentiment_counts[0])
    print("Neutral (1):", sentiment_counts[1])
    print("Positive (2):", sentiment_counts[2])
    
    # Create balanced dataset
    balanced_reviews = create_balanced_dataset(reviews, samples_per_class=142)  # Using the size of the smallest class
    
    # Display new distribution
    new_counts = balanced_reviews['sentiment'].value_counts().sort_index()
    print("\nBalanced Dataset Distribution:")
    print("-" * 30)
    print("Negative (0):", new_counts[0])
    print("Neutral (1):", new_counts[1])
    print("Positive (2):", new_counts[2])
    
    # Display some examples
    print("\nExample reviews from balanced dataset:")
    print("-" * 30)
    for _, row in balanced_reviews.head(6).iterrows():
        sentiment_name = ['Negative', 'Neutral', 'Positive'][row['sentiment']]
        print(f"Rating: {row['overall']} -> Sentiment: {sentiment_name}")
        print(f"Review: {row['reviewText'][:100]}...\n")
    
    # Save the balanced dataset
    output_file = "balanced_reviews.csv"
    print(f"\nSaving balanced dataset to {output_file}...")
    balanced_reviews.to_csv(output_file, index=False)
    print("Conversion completed successfully!")

if __name__ == "__main__":
    main() 