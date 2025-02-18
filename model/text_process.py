import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text(text):
    # Add spaces around punctuation marks
    text = re.sub(r'([.,!?])', r' \1 ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def clean_text(text):
    # Convert to lowercase
    text = str(text).lower()
    
    # Preprocess punctuation
    text = preprocess_text(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s.,!?]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in words])
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words])
    
    # Remove very short words (length < 2)
    words = text.split()
    text = ' '.join([word for word in words if len(word) > 2])
    
    return text

def main():
    try:
        # Load the raw reviews
        reviews = pd.read_csv("processed_data/raw_reviews.csv")
        
        # Clean the text
        reviews['cleaned_text'] = reviews['reviewText'].apply(clean_text)
        
        # Save processed data
        output_file = "processed_data/cleaned_reviews.csv"
        reviews.to_csv(output_file, index=False)
        
        print("Text processing completed!")
        
    except FileNotFoundError:
        print("Error: Could not find input file. Make sure raw_reviews.csv is in the processed_data folder.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()