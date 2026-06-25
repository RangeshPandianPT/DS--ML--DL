import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import pickle

def main():
    print("Loading Stock data...")
    # Read the data, skip the first row if it contains a weird unnamed column header mismatch
    data = pd.read_csv("stock_data.csv")
    
    # Selecting the relevant features: Open, High, Low, Volume
    # Predicting: Close
    
    # Make sure we don't have NaNs
    data = data.dropna(subset=['Open', 'High', 'Low', 'Volume', 'Close'])
    
    X = data[['Open', 'High', 'Low', 'Volume']].values
    y = data['Close'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Stock Prediction regression model...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"Model trained successfully. Test R^2 Score: {score:.4f}")
    
    model_filename = 'stock_prediction_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    main()
