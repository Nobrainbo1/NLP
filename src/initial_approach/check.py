import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

reviews = pd.read_csv("amazon_reviews.csv", encoding='utf-8')
selected_reviews = reviews[['reviewText', 'overall']]

selected_reviews['reviewText'] = selected_reviews['reviewText'].fillna('')


# Plot the distribution of scores
score_counts = selected_reviews['overall'].value_counts().sort_index()
plt.figure(figsize=(8, 6))
score_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Review Scores')
plt.xlabel('Score')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.show()
