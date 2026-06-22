import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

def main():
    print("Loading SUV data...")
    # Load dataset
    suv_data = pd.read_csv("suv_data.csv")
    
    # Features: Age, EstimatedSalary (columns 2 and 3)
    # Target: Purchased (column 4)
    X = suv_data.iloc[:, [2, 3]].values
    y = suv_data.iloc[:, 4].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    print("Training model pipeline...")
    # Create pipeline with scaling and logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    score = pipeline.score(X_test, y_test)
    print(f"Model trained successfully. Test Accuracy: {score:.4f}")
    
    # Save the pipeline
    model_filename = 'suv_prediction_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    main()
