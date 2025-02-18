import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
import numpy as np

def create_distribution_plot():
    """Create sentiment distribution plot"""
    # Load the data
    reviews = pd.read_csv("processed_data/raw_reviews.csv")
    
    # Create distribution plot
    plt.figure(figsize=(10, 6))
    sentiment_counts = reviews['sentiment'].value_counts().sort_index()
    
    # Plot
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    plt.bar(['Negative', 'Neutral', 'Positive'], sentiment_counts, color=colors)
    plt.title('Distribution of Sentiments in Dataset')
    plt.ylabel('Number of Reviews')
    
    # Add value labels on top of bars
    for i, v in enumerate(sentiment_counts):
        plt.text(i, v + 1, str(v), ha='center')
    
    plt.savefig('progress_report/sentiment_distribution.png')
    plt.close()

def create_model_performance_plot():
    """Create model performance visualization"""
    # Load the model and test data
    with open('processed_data/tfidf_features.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('processed_data/naive_bayes_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Get predictions
    y_pred = model.predict(data['X'])
    
    # Create confusion matrix
    cm = confusion_matrix(data['y'], y_pred)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('progress_report/confusion_matrix.png')
    plt.close()

def create_text_length_distribution():
    """Create text length distribution plot"""
    # Load cleaned reviews
    reviews = pd.read_csv("processed_data/cleaned_reviews.csv")
    
    # Calculate text lengths
    reviews['text_length'] = reviews['cleaned_text'].str.len()
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='sentiment', y='text_length', 
                data=reviews, 
                palette=['#ff9999', '#66b3ff', '#99ff99'])
    plt.title('Review Length Distribution by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Text Length')
    plt.xticks([0, 1, 2], ['Negative', 'Neutral', 'Positive'])
    plt.savefig('progress_report/text_length_distribution.png')
    plt.close()

def generate_statistics():
    """Generate and save key statistics"""
    # Load data
    reviews = pd.read_csv("processed_data/cleaned_reviews.csv")
    
    # Calculate statistics
    stats = {
        'Total Reviews': len(reviews),
        'Average Text Length': int(reviews['cleaned_text'].str.len().mean()),
        'Median Text Length': int(reviews['cleaned_text'].str.len().median()),
        'Sentiment Distribution': reviews['sentiment'].value_counts().to_dict()
    }
    
    # Save statistics
    with open('progress_report/statistics.txt', 'w') as f:
        f.write("Dataset Statistics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Reviews: {stats['Total Reviews']}\n")
        f.write(f"Average Text Length: {stats['Average Text Length']} characters\n")
        f.write(f"Median Text Length: {stats['Median Text Length']} characters\n")
        f.write("\nSentiment Distribution:\n")
        for sentiment, count in stats['Sentiment Distribution'].items():
            sentiment_name = ['Negative', 'Neutral', 'Positive'][sentiment]
            f.write(f"{sentiment_name}: {count} reviews\n")

def main():
    try:
        # Create progress_report directory if it doesn't exist
        import os
        if not os.path.exists('progress_report'):
            os.makedirs('progress_report')
        
        print("Generating visualizations and statistics...")
        
        # Create visualizations
        create_distribution_plot()
        create_model_performance_plot()
        create_text_length_distribution()
        
        # Generate statistics
        generate_statistics()
        
        print("\nProgress report generated successfully!")
        print("Check the 'progress_report' folder for visualizations and statistics.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 