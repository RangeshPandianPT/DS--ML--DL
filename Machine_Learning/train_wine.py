import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import pickle

def main():
    print("Loading Wine Quality data...")
    data = pd.read_csv("WineQT.csv")
    
    # Drop Id column
    if 'Id' in data.columns:
        data = data.drop('Id', axis=1)
        
    X = data.drop('quality', axis=1).values
    y = data['quality'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Wine Quality regression model...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"Model trained successfully. Test R^2 Score: {score:.4f}")
    
    model_filename = 'wine_quality_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    main()
