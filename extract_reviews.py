import pandas as pd

def main():
    try:
        # Load the data
        print("Loading the dataset...")
        reviews = pd.read_csv("amazon_reviews.csv", encoding='utf-8')
        
        # Remove empty reviews
        reviews = reviews[reviews['reviewText'].notna()].reset_index(drop=True)
        
        # Create a new dataframe with just the review text and a blank column for manual labels
        print("Preparing dataset for manual labeling...")
        manual_labeling_df = pd.DataFrame({
            'review_text': reviews['reviewText'],
            'sentiment': ''  # Empty column for manual labels
        })
        
        # Sample 1000 reviews randomly (or adjust the number as needed)
        sampled_reviews = manual_labeling_df.sample(n=1000, random_state=42).reset_index(drop=True)
        
        # Save to CSV
        output_file = "reviews_for_labeling.csv"
        print(f"\nSaving {len(sampled_reviews)} reviews to {output_file}...")
        sampled_reviews.to_csv(output_file, index=False, encoding='utf-8')
        
        print("\nFile created successfully!")
        print("\nInstructions for labeling:")
        print("1. Open 'reviews_for_labeling.csv' in Excel or any spreadsheet software")
        print("2. For each review in the 'review_text' column, fill in the 'sentiment' column with:")
        print("   0 for Negative")
        print("   1 for Neutral")
        print("   2 for Positive")
        print("\nTips for labeling:")
        print("- You can use Excel's filters and sorting to help with the labeling process")
        print("- Save your progress regularly")
        print("- You can label in batches to make it more manageable")
        
    except FileNotFoundError:
        print("Error: Could not find 'amazon_reviews.csv'. Make sure it's in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 