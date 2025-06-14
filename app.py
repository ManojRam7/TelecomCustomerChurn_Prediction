import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Load the trained model, scaler, and other necessary objects
# Make sure these paths are correct based on where you save your files
try:
    best_model = joblib.load('best_churn_model.pkl')
    # Assuming you saved the scaler as well
    # scaler = joblib.load('scaler.pkl')
    # You might need to load other objects here if your predict_churn function requires them
    # For example, if you saved the list of original numerical columns
    # original_numerical_cols = joblib.load('original_numerical_cols.pkl')
    # If your median imputation relies on the original df, you might need to handle that differently in deployment
    # Perhaps save the median value itself: median_total_charges = joblib.load('median_total_charges.pkl')

    # For demonstration, let's assume the scaler is available and we saved the original X columns and numerical columns
    # In a real deployment, you would save these explicitly
    # If running this after the previous steps in the same Colab session, these might be available
    if 'scaler' in locals() and 'X' in locals() and 'numerical_cols_before_encoding' in locals():
         print("Loaded scaler, X columns, and numerical columns from current session.")
         scaler_loaded = scaler # Use the scaler from the current session
         original_X_cols_loaded = X.columns # Use columns from the current session's X
         original_numerical_cols_loaded = numerical_cols_before_encoding # Use the list from the current session
         # Assuming you also need the median TotalCharges if there were missing values in the original data
         # You should have calculated and saved this median during your training pipeline
         # For this example, let's assume you have a variable `median_total_charges` from your training
         if 'df' in locals() and 'TotalCharges' in df.columns:
             median_total_charges_loaded = df['TotalCharges'].median()
         else:
              median_total_charges_loaded = None # Handle case where median is not available
              st.warning("Median TotalCharges not available. Missing TotalCharges will not be imputed.")

    else:
        st.error("Required preprocessing objects (scaler, original X columns, original numerical columns) not found. Please ensure they are saved and loaded correctly.")
        st.stop() # Stop the app if essential objects are missing

except FileNotFoundError:
    st.error("Model file or other required files not found. Please ensure 'best_churn_model.pkl' and other necessary files are in the correct directory.")
    st.stop() # Stop the app if the model file is not found

# Define the prediction function (similar to Step 14 in your notebook)
def predict_churn(new_data_original, model, scaler, median_total_charges, original_X_cols, original_numerical_cols):
    """
    Predicts churn for new customer data.

    Args:
        new_data_original (pd.DataFrame): DataFrame with new customer data
                                           in the original feature format.
        model: Trained machine learning model.
        scaler: Fitted StandardScaler object used for training data.
        median_total_charges (float): Median TotalCharges from the training data for imputation.
        original_X_cols (list): List of column names of the training data (X).
        original_numerical_cols (list): List of numerical column names from the original df
                                         before encoding.

    Returns:
        tuple: (prediction, prediction_probability) or (None, None) if error occurs.
    """
    # Make a copy to avoid modifying the original input DataFrame
    new_data_processed = new_data_original.copy()

    # Apply the same preprocessing steps as training data

    # Handle 'TotalCharges'
    if 'TotalCharges' in new_data_processed.columns:
        new_data_processed['TotalCharges'] = pd.to_numeric(new_data_processed['TotalCharges'], errors='coerce')
        if median_total_charges is not None:
             new_data_processed['TotalCharges'].fillna(median_total_charges, inplace=True)
        # Note: In a real deployment, you'd need a robust way to get the median if not saved.
    else:
        st.warning("Warning: 'TotalCharges' column not found in new data for prediction.")


    # Create new features
    if 'tenure' in new_data_processed.columns:
        new_data_processed['tenure_years'] = new_data_processed['tenure'] / 12
    if 'InternetService' in new_data_processed.columns:
        new_data_processed['has_internet'] = new_data_processed['InternetService'].apply(lambda x: 0 if x in ['No', 'No internet service'] else 1)


    # Handle categorical variables - One-Hot Encoding
    categorical_cols_new = [col for col in new_data_processed.columns if new_data_processed[col].dtype == 'object' and col != 'customerID']
    new_data_encoded = pd.get_dummies(new_data_processed, columns=categorical_cols_new, drop_first=True)

    # Reindex to match training columns
    new_data_aligned = new_data_encoded.reindex(columns=original_X_cols, fill_value=0)


    # Apply scaling
    if scaler is not None and original_numerical_cols is not None:
         numerical_cols_after_encoding_new = [col for col in new_data_aligned.columns if col in original_numerical_cols]
         if numerical_cols_after_encoding_new: # Check if there are numerical columns to scale
            try:
                new_data_aligned[numerical_cols_after_encoding_new] = scaler.transform(new_data_aligned[numerical_cols_after_encoding_new])
            except ValueError as e:
                 st.error(f"Error during scaling: {e}. This might be due to inconsistent data or missing numerical columns.")
                 return None, None
         else:
            st.warning("Warning: No numerical columns found in new data to scale during prediction.")
    else:
        st.warning("Warning: Scaler or original numerical columns not available for scaling during prediction.")


    # Ensure column order and names match before prediction
    if list(new_data_aligned.columns) == list(original_X_cols):
        # Make prediction
        prediction = model.predict(new_data_aligned)
        prediction_proba = model.predict_proba(new_data_aligned)[:, 1]

        return prediction[0], prediction_proba[0]
    else:
        st.error("Error: Columns of new processed data do not match training data columns. Cannot make prediction.")
        st.write("New data columns:", list(new_data_aligned.columns))
        st.write("Training data columns:", list(original_X_cols))
        return None, None

# --- Streamlit App Interface ---

st.title("Telco Customer Churn Prediction")

st.write("Enter customer details to predict if they will churn.")

# Input fields for customer data
# You'll need to create input widgets for each feature in your original dataset
# Here are some examples:

st.sidebar.header("Customer Information")

gender = st.sidebar.radio("Gender", ['Female', 'Male'])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
partner = st.sidebar.radio("Partner", ['Yes', 'No'])
dependents = st.sidebar.radio("Dependents", ['Yes', 'No'])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 1)
phone_service = st.sidebar.radio("Phone Service", ['Yes', 'No'])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
internet_service = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
online_security = st.sidebar.selectbox("Online Security", ['No internet service', 'No', 'Yes'])
online_backup = st.sidebar.selectbox("Online Backup", ['No internet service', 'No', 'Yes'])
device_protection = st.sidebar.selectbox("Device Protection", ['No internet service', 'No', 'Yes'])
tech_support = st.sidebar.selectbox("Tech Support", ['No internet service', 'No', 'Yes'])
streaming_tv = st.sidebar.selectbox("Streaming TV", ['No internet service', 'No', 'Yes'])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ['No internet service', 'No', 'Yes'])
contract = st.sidebar.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.sidebar.radio("Paperless Billing", ['Yes', 'No'])
payment_method = st.sidebar.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=1000.0)

# Create a DataFrame from the input values
# This DataFrame should have the same structure as your original input
