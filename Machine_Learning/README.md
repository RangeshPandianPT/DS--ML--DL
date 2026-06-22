# Machine Learning Projects

This folder contains a collection of Machine Learning algorithms, exploratory data analysis, and end-to-end predictive modeling projects.

## Contents

### 1. Pre-Trained Models 🧠
We have exported machine learning models ready for deployment:
* `height_intelligence.pkl` - A Linear Regression model that predicts a person's weight (in kg) based on their height (in cm).
* `gender_classification_intelligence.pkl` - A Logistic Regression model that classifies gender based on height and weight.
* `suv_prediction_model.pkl` - A Logistic Regression model (trained via `train_suv.py`) that predicts SUV purchase behavior based on age and salary.
> **Note:** A Streamlit UI (`Machine_Learning_App.py`) is provided to interact with these models!

### 2. Jupyter Notebooks 📓
The repository includes several notebooks covering different ML concepts:
* **Exploratory Data Analysis.ipynb**: Comprehensive EDA techniques.
* **Feature_Engineering/**: Techniques for feature selection, transformation, and creation.
* **Basics.ipynb**: Fundamental concepts in Machine Learning.
* **Algorithms Implementation**:
  * `Logistic_Regression.ipynb`
  * `K-Mean_Clustering.ipynb`
* **Evaluation & Best Practices**:
  * `AUC ROC Curve.ipynb`
  * `Confusion Matrix.ipynb`
  * `Cross Validation.ipynb`
  * `Regularization.ipynb`
* **Projects**:
  * `Height_Prediction.ipynb`
  * `SUV_Prediction.ipynb`
  * `Iris_Dataset_Analysis_Project.ipynb`
  * `Time Series Analysis & Visualization.ipynb`

### 3. Datasets 📊
* `weight-height.csv` - Used for height/weight prediction.
* `suv_data.csv` - Used for SUV purchase prediction.
* `Iris.csv` - Classic Iris dataset.
* `WineQT.csv` - Wine quality dataset.
* `stock_data.csv` - Stock market dataset.
* `Universities.csv` - University statistics dataset.

## How to Run the Web App
To interact with the pre-trained models, run the Streamlit application:

```bash
cd Machine_Learning
streamlit run Machine_Learning_App.py
```
