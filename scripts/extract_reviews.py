import pandas as pd
import os
import numpy as np

def extract_and_mix_reviews():
    # Create output directory if it doesn't exist
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    
    try:
        # Read the first CSV file
        df1 = pd.read_csv('data/archive/7817_1.csv')
        reviews1 = df1['reviews.text'].dropna().tolist()
        
        # Read the second CSV file
        df2 = pd.read_csv('data/archive copy/amazon_reviews.csv')
        reviews2 = df2['reviewText'].dropna().tolist()
        
        # Combine reviews
        all_reviews = reviews1 + reviews2
        
        # Create a DataFrame with the reviews
        combined_df = pd.DataFrame({
            'review_text': all_reviews,
            'sentiment': [''] * len(all_reviews)  # Empty column for manual labeling
        })
        
        # Shuffle the DataFrame randomly
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save to new CSV file
        output_path = 'C:\Uni_year3_term2\NLP\Project\data'
        combined_df.to_csv(output_path, index=False)
        print(f"Successfully extracted and mixed {len(all_reviews)} reviews to {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    extract_and_mix_reviews()