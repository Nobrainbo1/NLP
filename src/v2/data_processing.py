import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from textblob import TextBlob
import emoji

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def normalize_text_length(text, target_length=200):
    """
    Normalize text length while preserving sentiment-carrying words
    """
    words = text.split()
    
    if len(words) > target_length:
        # Keep the first 75% and last 25% of words to preserve both context and conclusion
        split_point = int(target_length * 0.75)
        beginning = words[:split_point]
        end = words[-(target_length - split_point):]
        return ' '.join(beginning + end)
    return ' '.join(words)  # Keep short texts as is

def convert_emojis(text):
    """Convert emojis to text representation"""
    return emoji.demojize(text).replace('_', ' ')

def preserve_sentiment_markers(text):
    """Preserve common sentiment markers"""
    # Add spaces around sentiment punctuation
    text = re.sub(r'([!?.]){2,}', r' \1\1\1 ', text)  # Convert multiple punctuation to triple
    text = re.sub(r'([!?.])', r' \1 ', text)
    
    # Preserve common sentiment-carrying symbols
    text = text.replace(':)', ' happy_smile ')
    text = text.replace(':(', ' sad_smile ')
    text = text.replace('❤', ' heart ')
    return text

def enhanced_clean_text(text):
    """Enhanced text cleaning pipeline with sentiment preservation"""
    # Convert to lowercase but preserve sentiment markers first
    text = preserve_sentiment_markers(str(text).lower())
    
    # Convert emojis to text
    text = convert_emojis(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters but preserve sentiment punctuation
    text = re.sub(r'[^a-zA-Z\s.,!?]', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    
    # Only lemmatize words that aren't sentiment markers
    sentiment_markers = {'happy_smile', 'sad_smile', 'heart', '!!!', '??', '...'}
    text = ' '.join([word if word in sentiment_markers else lemmatizer.lemmatize(word) for word in words])
    
    # Remove stopwords but preserve negations
    stop_words = set(stopwords.words('english')) - {'no', 'not', 'nor', 'never', 'hardly', 'barely'}
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words or word in {'no', 'not', 'nor', 'never'}])
    
    # Remove very short words except sentiment markers
    text = ' '.join([word for word in text.split() if len(word) > 2 or word in sentiment_markers])
    
    # Normalize length while preserving sentiment
    text = normalize_text_length(text)
    
    return text

def analyze_text_statistics(texts):
    """Analyze text statistics for reporting"""
    lengths = [len(text.split()) for text in texts]
    return {
        'mean_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths)
    }

def main():
    try:
        # Load the raw reviews
        reviews = pd.read_csv("processed_data/sentiment_dataset.csv")
        
        # Apply enhanced cleaning
        reviews['cleaned_text'] = reviews['text'].apply(enhanced_clean_text)
        
        # Analyze and print statistics
        stats = analyze_text_statistics(reviews['cleaned_text'])
        print("\nText Statistics after processing:")
        print(f"Mean length: {stats['mean_length']:.1f} words")
        print(f"Median length: {stats['median_length']:.1f} words")
        print(f"Length range: {stats['min_length']} to {stats['max_length']} words")
        
        # Save processed data
        output_file = "processed_data/enhanced_cleaned_reviews.csv"
        reviews.to_csv(output_file, index=False)
        
        print("\nEnhanced text processing completed!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()