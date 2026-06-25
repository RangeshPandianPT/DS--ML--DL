import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pickle

def main():
    print("Loading Iris data...")
    data = pd.read_csv("Iris.csv")
    
    # Features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
    X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
    y = data['Species'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Iris classification model...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"Model trained successfully. Test Accuracy: {score:.4f}")
    
    model_filename = 'iris_classification_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    main()
