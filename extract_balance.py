import pandas as pd

def main():
    try:
        # Load the balanced reviews
        print("Loading balanced reviews dataset...")
        reviews = pd.read_csv("balanced_reviews.csv")
        
        # Select only reviewText and sentiment columns
        simple_reviews = reviews[['reviewText', 'sentiment']]
        
        # Save to a new file in the data folder
        output_file = "processed_data/raw_reviews.csv"
        print(f"\nSaving simplified dataset to {output_file}...")
        simple_reviews.to_csv(output_file, index=False, encoding='utf-8')
        
        print("\nFile created successfully!")
        
    except FileNotFoundError:
        print("Error: Could not find 'balanced_reviews.csv'. Make sure it's in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 