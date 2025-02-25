import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pickle
import random
import re
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')

def create_word_embeddings(texts, max_words=100000, max_len=200):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    
    with open('processed_data/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
        
    return padded_sequences, tokenizer

def create_lstm_model(vocab_size, embedding_dim=100, max_len=200, embedding_matrix=None):
    model = Sequential([
        # Embedding layer with proper initialization
        Embedding(vocab_size, embedding_dim*2, 
                 weights=[embedding_matrix] if embedding_matrix is not None else None,
                 trainable=False if embedding_matrix is not None else True,
                 input_length=max_len,
                 mask_zero=True),  # Add masking for variable length
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),  # Add batch normalization
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def get_synonyms(word):
    """Get synonyms for a word using WordNet"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.add(lemma.name())
    return list(synonyms)

def mixup_sequences(seq1, seq2, label1, label2, alpha=0.2):
    """
    Apply mixup augmentation to sequences and their labels
    λ ~ Beta(α, α)
    """
    lambda_param = np.random.beta(alpha, alpha)
    lambda_param = max(lambda_param, 1 - lambda_param)
    
    mixed_seq = lambda_param * seq1 + (1 - lambda_param) * seq2
    mixed_label = lambda_param * label1 + (1 - lambda_param) * label2
    
    return mixed_seq, mixed_label

def augment_text(text, method='synonym'):
    words = text.split()
    if len(words) < 5:
        return text
    
    # Expanded sentiment words list
    negative_words = {'hate', 'bad', 'terrible', 'awful', 'worst', 'poor', 'horrible'}
    positive_words = {'great', 'good', 'awesome', 'excellent', 'best', 'love'}
    modifiers = {'not', 'no', 'never', 'hardly', 'barely', "doesn't", "don't", "didn't"}
    
    if method == 'synonym':
        num_to_replace = max(1, int(len(words) * 0.2))
        replace_indices = random.sample(range(len(words)), num_to_replace)
        
        for idx in replace_indices:
            word = words[idx]
            # Don't replace sentiment-critical words
            if word not in negative_words and word not in positive_words and word not in modifiers:
                synonyms = get_synonyms(word)
                if synonyms:
                    words[idx] = random.choice(synonyms)
        
        return ' '.join(words)
    
    elif method == 'reorder':
        # Keep negative phrases together
        segments = re.split(r'[.,!?]', text)
        segments = [s.strip() for s in segments if s.strip()]
        
        if len(segments) > 1:
            # Keep negative segments in their original position
            neg_segments = [s for s in segments if any(word in s.lower() for word in negative_words)]
            other_segments = [s for s in segments if s not in neg_segments]
            random.shuffle(other_segments)
            return ' . '.join(neg_segments + other_segments)
    
    return text

def create_augmented_dataset(df):
    """Create augmented dataset using multiple techniques"""
    augmented_data = []
    
    # Original data
    for _, row in df.iterrows():
        augmented_data.append({
            'cleaned_text': row['cleaned_text'],
            'sentiment': row['sentiment']
        })
        
        # Add synonym-based augmentation
        augmented_data.append({
            'cleaned_text': augment_text(row['cleaned_text'], method='synonym'),
            'sentiment': row['sentiment']
        })
        
        # Add reordering augmentation for longer texts
        if len(row['cleaned_text'].split()) > 10:
            augmented_data.append({
                'cleaned_text': augment_text(row['cleaned_text'], method='reorder'),
                'sentiment': row['sentiment']
            })
    
    return pd.DataFrame(augmented_data)

def load_glove_embeddings(word_index, embedding_dim=100):
    embeddings_index = {}
    try:
        with open('glove.6B.100d.txt', encoding='utf-8') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                embeddings_index[word] = coefs
    except FileNotFoundError:
        print("Warning: GloVe embeddings file not found. Using random embeddings instead.")
        return None
    
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix

def main():
    try:
        # Load the processed data
        df = pd.read_csv('processed_data/enhanced_cleaned_reviews.csv')
        
        # Create augmented dataset
        augmented_df = create_augmented_dataset(df)
        
        # Create word embeddings
        X_padded, tokenizer = create_word_embeddings(augmented_df['cleaned_text'])
        y = augmented_df['sentiment'].values
        
        # Split the data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_padded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply mixup augmentation to training data
        mixed_X = []
        mixed_y = []
        num_mixup = len(X_train) // 2
        
        for _ in range(num_mixup):
            idx1, idx2 = np.random.randint(0, len(X_train), 2)
            mixed_seq, mixed_label = mixup_sequences(
                X_train[idx1], X_train[idx2],
                y_train[idx1], y_train[idx2]
            )
            mixed_X.append(mixed_seq)
            mixed_y.append(mixed_label)
        
        # Combine original and mixup data
        X_train = np.vstack([X_train, np.array(mixed_X)])
        y_train = np.concatenate([y_train, np.array(mixed_y)])
        
        # Try to load pre-trained embeddings
        embedding_matrix = load_glove_embeddings(tokenizer.word_index, embedding_dim=200)
        
        # Create and train the model
        vocab_size = len(tokenizer.word_index) + 1
        model = create_lstm_model(vocab_size, embedding_dim=200, embedding_matrix=embedding_matrix)
        
        # Early stopping with more patience
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            min_delta=0.001
        )
        
        # Train with smaller batch size and more epochs
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=8,
            validation_split=0.2,
            callbacks=[early_stopping]
        )
        
        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"\nTest accuracy: {accuracy:.2%}")
        
        # Save the model
        model.save('processed_data/lstm_model.h5')
        
        # Test with sample reviews
        sample_reviews = [
            "This product is terrible",
            "It's okay, nothing special",
            "Amazing product, I love it!"
        ]
        
        # Process sample reviews
        sample_sequences = tokenizer.texts_to_sequences(sample_reviews)
        sample_padded = pad_sequences(sample_sequences, maxlen=200, padding='post')
        
        # Get predictions
        predictions = model.predict(sample_padded)
        
        print("\nSample Predictions:")
        print("-"*50)
        for review, pred in zip(sample_reviews, predictions):
            sentiment = ['Negative', 'Neutral', 'Positive'][np.argmax(pred)]
            print(f"\nReview: '{review}'")
            print(f"Predicted sentiment: {sentiment}")
            print(f"Confidence scores:")
            print(f"  Negative: {pred[0]:.2%}")
            print(f"  Neutral:  {pred[1]:.2%}")
            print(f"  Positive: {pred[2]:.2%}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 