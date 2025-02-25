import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def main():
    try:
        # Load the features
        with open('processed_data/tfidf_features.pkl', 'rb') as f:
            data = pickle.load(f)
        
        X = data['X']
        y = data['y']
        vectorizer = data['vectorizer']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train and evaluate Logistic Regression
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\nLogistic Regression Results:")
        print(f"Overall Accuracy: {accuracy:.2%}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                target_names=['Negative', 'Neutral', 'Positive']))
        
        # Save the model
        with open('processed_data/logistic_regression_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        print("\nModel saved as logistic_regression_model.pkl")
        
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
        for review, pred, prob in zip(sample_reviews, predictions, probabilities):
            sentiment = ['Negative', 'Neutral', 'Positive'][pred]
            print(f"\nReview: '{review}'")
            print(f"Predicted sentiment: {sentiment}")
            print(f"Confidence scores: Negative: {prob[0]:.2%}, Neutral: {prob[1]:.2%}, Positive: {prob[2]:.2%}")
        
        print("\nModel training completed!")
        
    except FileNotFoundError:
        print("Error: Could not find input file. Make sure tfidf_features.pkl is in the processed_data folder.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()