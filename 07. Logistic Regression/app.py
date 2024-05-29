import streamlit as st
import joblib
import numpy as np

# Load the trained logistic regression model
model = joblib.load('logistic_regression_model.pkl')

st.title('Logistic Regression Model Deployment')

# Create inputs for user to enter feature values
st.header('Input Features')
feature1 = st.number_input('Feature 1')
feature2 = st.number_input('Feature 2')
# Add more inputs as necessary

# Make prediction
if st.button('Predict'):
    features = np.array([[feature1, feature2]])  # Add more features as necessary
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    
    st.write(f'Prediction: {prediction[0]}')
    st.write(f'Probability: {probability[0]}')

st.write("This is a simple logistic regression model deployed using Streamlit.")
