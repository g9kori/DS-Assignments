# Logistic Regression Model Deployment with Streamlit

This project demonstrates how to deploy a logistic regression model using Streamlit. The deployment is done locally.

## Prerequisites

- Python 3.x
- Web browser

## Instructions

### 1. Create a Virtual Environment
To ensure a clean working environment, it's recommended to create a virtual environment. Use the following commands:
python -m venv env

### 2. Activate the virtual environment
#### On Windows
.\env\Scripts\activate
#### On macOS/Linux
source env/bin/activate

### 3. Install Required Packages
pip install streamlit joblib pandas numpy scikit-learn

### 4. Ensure Files are in the Same Directory
- app.py (this Streamlit script)
- logistic_regression_model.pkl (the trained logistic regression model)
- .pkl file (the saved scaler)
- Titanic_train.csv and Titanic_test.csv (the Titanic datasets)

### 5. Run the Streamlit App
streamlit run app.py

### 6. Open Web Browser
After running the above code, if it doesn't auotomatically redirects to browser then navigate to http://localhost:8501/ to access the Streamlit app.
