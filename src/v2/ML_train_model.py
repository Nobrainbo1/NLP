import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def main():
    try:
        # Load the features
        with open('processed_data/tfidf_featuresv2.pkl', 'rb') as f:
            data = pickle.load(f)
        
        X = data['X']
        y = data['y']
        vectorizer = data['vectorizer']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*50)
        print("Naive Bayes Results:")
        print("="*50)
        print(f"Overall Accuracy: {accuracy:.2%}")
        print("\nDetailed Classification Report:")
        print("-"*50)
        print(classification_report(y_test, y_pred, 
                                target_names=['Negative', 'Neutral', 'Positive']))
        
        # Save the model
        with open('processed_data/naive_bayes_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Test with some sample reviews
        sample_reviews = [
            "This product is terrible",
            "It's okay, nothing special",
            "Amazing product, I love it!"
        ]
        
        # Transform the samples
        sample_vectors = vectorizer.transform(sample_reviews)
        predictions = model.predict(sample_vectors)
        probabilities = model.predict_proba(sample_vectors)
        
        print("\nSample Predictions:")
        print("-"*50)
        for review, pred, prob in zip(sample_reviews, predictions, probabilities):
            sentiment = ['Negative', 'Neutral', 'Positive'][pred]
            print(f"\nReview: '{review}'")
            print(f"Predicted sentiment: {sentiment}")
            print(f"Confidence scores:")
            print(f"  Negative: {prob[0]:.2%}")
            print(f"  Neutral:  {prob[1]:.2%}")
            print(f"  Positive: {prob[2]:.2%}")
        
        print("\nModel training completed!")
        
    except FileNotFoundError:
        print("Error: Could not find input file. Make sure tfidf_features.pkl is in the processed_data folder.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 