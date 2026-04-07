import pytest
import pandas as pd
from loan_default_predictor import model, preprocessor, train_df, target

def test_train_df_columns():
    """Verify that the training data has all the expected columns."""
    expected_cols = [
        'person_age', 'person_income', 'person_home_ownership', 'person_emp_length', 
        'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 
        'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length',
        'loan_status'
    ]
    for col in expected_cols:
        assert col in train_df.columns

def test_model_pipeline_predict():
    """Test that the trained model can make a prediction on a sample row."""
    # Create a single row of data matching the expected features
    sample_row = pd.DataFrame([{
        'person_age': 25,
        'person_income': 50000,
        'person_home_ownership': 'RENT',
        'person_emp_length': 2.0,
        'loan_intent': 'EDUCATION',
        'loan_grade': 'A',
        'loan_amnt': 5000,
        'loan_int_rate': 10.5,
        'loan_percent_income': 0.1,
        'cb_person_default_on_file': 'N',
        'cb_person_cred_hist_length': 3
    }])
    
    prediction = model.predict(sample_row)
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]

def test_preprocessor_is_fitted():
    """Test if the preprocessor transform function works after fit."""
    sample_row = pd.DataFrame([{
        'person_age': 30,
        'person_income': 60000,
        'person_home_ownership': 'OWN',
        'person_emp_length': 5.0,
        'loan_intent': 'PERSONAL',
        'loan_grade': 'B',
        'loan_amnt': 7000,
        'loan_int_rate': 12.0,
        'loan_percent_income': 0.12,
        'cb_person_default_on_file': 'N',
        'cb_person_cred_hist_length': 5
    }])
    
    transformed = model.named_steps['preprocessor'].transform(sample_row)
    # The output should be a numpy array/sparse matrix with transformed features
    assert transformed.shape[0] == 1
    assert transformed.shape[1] > 0
