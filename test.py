import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize

# Download the punkt tokenizer if not already downloaded
nltk.download('punkt')

# Load and preprocess the data
reviews = pd.read_csv("amazon_reviews.csv", encoding='utf-8')
selected_reviews = reviews[['reviewText', 'overall']]

# Convert ratings to sentiment classes
def convert_to_sentiment(rating):
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

# Create sentiment labels
selected_reviews['sentiment'] = selected_reviews['overall'].apply(convert_to_sentiment)

# Fill missing values
selected_reviews.loc[:, 'reviewText'] = selected_reviews['reviewText'].fillna('')

# Tokenize the text using nltk
selected_reviews.loc[:, 'reviewText'] = selected_reviews['reviewText'].apply(word_tokenize)

# Join the tokens back into a string for TfidfVectorizer
selected_reviews.loc[:, 'reviewText'] = selected_reviews['reviewText'].apply(lambda x: ' '.join(x))

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for faster processing
X = vectorizer.fit_transform(selected_reviews['reviewText'])
y = selected_reviews['sentiment']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0, 
                          target_names=['Negative', 'Neutral', 'Positive']))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()

# Test the model with sample reviews
test_reviews = [
    "This product is terrible, I hate it",
    "It's okay, nothing special",
    "Amazing product, I love it!"
]

# Transform and predict test reviews
test_vectors = vectorizer.transform(test_reviews)
predictions = model.predict(test_vectors)

print("\nSample Predictions:")
sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
for review, pred in zip(test_reviews, predictions):
    print(f"Review: '{review}'")
    print(f"Predicted sentiment: {sentiment_map[pred]}\n")


