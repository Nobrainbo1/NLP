import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import emoji
import string
import contractions

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Define sentiment-critical words that should never be removed
SENTIMENT_WORDS = {
    # Negative sentiment words
    'bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing', 'disappointed', 
    'useless', 'waste', 'problem', 'issue', 'defective', 'broken', 'fail', 'failed',
    'fails', 'failure', 'worst', 'worse', 'sucks', 'suck', 'hate', 'hated', 'difficult',
    'impossible', 'error', 'errors', 'faulty', 'cheap', 'expensive', 'overpriced',
    'slow', 'difficult', 'hard', 'frustrating', 'annoying', 'avoid', 'return',
    
    # Positive sentiment words
    'good', 'great', 'excellent', 'amazing', 'awesome', 'fantastic', 'wonderful',
    'perfect', 'best', 'better', 'love', 'loved', 'like', 'liked', 'recommend',
    'recommended', 'easy', 'simple', 'fast', 'quick', 'reliable', 'worth', 'happy',
    'pleased', 'satisfied', 'quality', 'valuable', 'impressive', 'superb', 'outstanding',
    'exceptional', 'brilliant', 'flawless', 'seamless', 'convenient', 'enjoy', 'enjoyed',
    
    # Neutral/modifier words
    'okay', 'ok', 'fine', 'average', 'decent', 'mediocre', 'acceptable', 'fair',
    'moderate', 'standard', 'normal', 'regular', 'ordinary', 'typical', 'usual',
    'common', 'alright', 'reasonable', 'adequate', 'sufficient', 'satisfactory',
    
    # Common short verbs that should be preserved (to prevent lemmatization issues)
    'does', 'did', 'do', 'has', 'had', 'have', 'is', 'was', 'were', 'am', 'are', 'be',
    'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would',
    'get', 'got', 'use', 'used', 'try', 'tried', 'see', 'saw', 'say', 'said', 'put'
}

# Define negation words that should be preserved
NEGATION_WORDS = {'no', 'not', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor', 'hardly', 'barely', 'scarcely', 'rarely'}

# Define sentiment punctuation to preserve
SENTIMENT_PUNCTUATION = {'!!!', '???', '...', '!?', '?!'}

# Define sentiment emoticons to preserve
SENTIMENT_EMOTICONS = {':)', ':(', ':D', ':P', ';)', ':-)', ':-(', ':-D', ':-P', ';-)'}

def expand_contractions(text):
    """Expand contractions like don't to do not"""
    return contractions.fix(text)

def handle_negations(text):
    """Join negation words with the following word using underscores"""
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        if i < len(words) - 1 and words[i].lower() in NEGATION_WORDS and words[i+1].lower() not in NEGATION_WORDS:
            # Join negation with the next word
            result.append(f"{words[i]}_{words[i+1]}")
            i += 2
        else:
            result.append(words[i])
            i += 1
    return ' '.join(result)

def convert_emojis(text):
    """Convert emojis to text representation with spaces"""
    return emoji.demojize(text).replace('_', ' ')

def preserve_sentiment_markers(text):
    """Preserve common sentiment markers"""
    # Add spaces around sentiment punctuation
    for punct in SENTIMENT_PUNCTUATION:
        text = text.replace(punct, f" {punct} ")
    
    # Preserve common sentiment-carrying emoticons
    for emoticon in SENTIMENT_EMOTICONS:
        text = text.replace(emoticon, f" {emoticon} ")
    
    return text

def advanced_clean_text(text, preserve_case=False):
    """Advanced text cleaning pipeline with better sentiment preservation"""
    # Handle None or empty strings
    if not text or pd.isna(text):
        return ""
    
    # Expand contractions first (don't -> do not)
    text = expand_contractions(str(text))
    
    # Preserve sentiment markers before any other processing
    text = preserve_sentiment_markers(text)
    
    # Convert emojis to text
    text = convert_emojis(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Handle negations (not good -> not_good)
    text = handle_negations(text)
    
    # Convert to lowercase if not preserving case
    if not preserve_case:
        text = text.lower()
    
    # Remove special characters but preserve sentiment punctuation and emoticons
    allowed_chars = set(string.ascii_letters + string.digits + ' ' + '.,!?_')
    allowed_tokens = SENTIMENT_PUNCTUATION.union(SENTIMENT_EMOTICONS)
    
    words = text.split()
    cleaned_words = []
    
    for word in words:
        if word in allowed_tokens:
            cleaned_words.append(word)
        else:
            # Keep only allowed characters in other words
            cleaned_word = ''.join(c for c in word if c in allowed_chars)
            if cleaned_word:
                cleaned_words.append(cleaned_word)
    
    text = ' '.join(cleaned_words)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Lemmatization with exceptions for sentiment words
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    
    # Only lemmatize words that aren't sentiment markers or negations
    sentiment_tokens = SENTIMENT_WORDS.union(NEGATION_WORDS).union(SENTIMENT_PUNCTUATION).union(SENTIMENT_EMOTICONS)
    text = ' '.join([word if word.lower() in sentiment_tokens or '_' in word 
                    else lemmatizer.lemmatize(word) for word in words])
    
    # Remove stopwords but preserve negations and sentiment words
    stop_words = set(stopwords.words('english')) - NEGATION_WORDS - SENTIMENT_WORDS
    words = text.split()
    text = ' '.join([word for word in words if word.lower() not in stop_words or 
                    word.lower() in NEGATION_WORDS or 
                    any(sentiment_word in word.lower() for sentiment_word in SENTIMENT_WORDS) or
                    '_' in word])
    
    # Remove very short words except sentiment markers and negations
    text = ' '.join([word for word in text.split() if len(word) > 2 or 
                    word.lower() in NEGATION_WORDS or 
                    word in SENTIMENT_PUNCTUATION or 
                    word in SENTIMENT_EMOTICONS or
                    '_' in word])
    
    return text

def normalize_text_length(text, target_length=300):
    """Normalize text length while preserving sentiment-carrying words"""
    words = text.split()
    
    if len(words) > target_length:
        # Keep the first 75% and last 25% of words to preserve both context and conclusion
        split_point = int(target_length * 0.75)
        beginning = words[:split_point]
        end = words[-(target_length - split_point):]
        return ' '.join(beginning + end)
    return ' '.join(words)  # Keep short texts as is

def process_reviews(df, text_column='review_text', preserve_case=False, normalize_length=True):
    """Process reviews with advanced cleaning"""
    # Apply advanced cleaning
    df['cleaned_text'] = df[text_column].apply(lambda x: advanced_clean_text(x, preserve_case))
    
    # Normalize text length if requested
    if normalize_length:
        df['cleaned_text'] = df['cleaned_text'].apply(normalize_text_length)
    
    return df

def analyze_text_statistics(texts):
    """Analyze text statistics for reporting"""
    lengths = [len(text.split()) for text in texts]
    return {
        'mean_length': sum(lengths) / len(lengths) if lengths else 0,
        'median_length': sorted(lengths)[len(lengths)//2] if lengths else 0,
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0
    }

def main():
    try:
        # Load the balanced reviews
        reviews = pd.read_csv("data/balanced_reviews_v2.csv")
        
        # Track initial count
        initial_count = len(reviews)
        print(f"\nInitial review count: {initial_count}")
        
        # Apply advanced cleaning
        processed_reviews = process_reviews(reviews)
        
        # Create a normalized version of cleaned text for better duplicate detection
        processed_reviews['cleaned_text_normalized'] = processed_reviews['cleaned_text'].str.lower().str.strip()
        
        # Remove duplicates based on normalized cleaned_text to ensure unique entries
        before_dedup = len(processed_reviews)
        processed_reviews = processed_reviews.drop_duplicates(subset=['cleaned_text_normalized'])
        after_dedup = len(processed_reviews)
        duplicates_removed = before_dedup - after_dedup
        
        # Remove the temporary normalization column
        processed_reviews = processed_reviews.drop(columns=['cleaned_text_normalized'])
        
        # Analyze and print statistics
        stats = analyze_text_statistics(processed_reviews['cleaned_text'])
        print("\nText Statistics after processing:")
        print(f"Mean length: {stats['mean_length']:.1f} words")
        print(f"Median length: {stats['median_length']:.1f} words")
        print(f"Length range: {stats['min_length']} to {stats['max_length']} words")
        print(f"Duplicates removed: {duplicates_removed} ({duplicates_removed/before_dedup*100:.1f}%)")
        print(f"Final review count: {after_dedup}")
        
        # Save processed data
        output_file = "processed_data/v3_cleaned_reviews.csv"
        processed_reviews.to_csv(output_file, index=False)
        
        print("\nAdvanced text processing completed!")
        print(f"Processed data saved to {output_file}")
        
        # Print sample comparisons
        print("\nSample Comparisons (Original vs Cleaned):")
        for i in range(min(3, len(processed_reviews))):
            print(f"\nOriginal: {processed_reviews[reviews.columns[0]].iloc[i][:100]}...")
            print(f"Cleaned:  {processed_reviews['cleaned_text'].iloc[i][:100]}...")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()