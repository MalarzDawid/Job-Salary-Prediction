# ####################
# # Imports libraries
# ####################

import xgboost as XGB
import pandas as pd
import pickle
import streamlit as st
import os
from sklearn.preprocessing import LabelEncoder
####################
# Custom functions
####################
def create_salary_range(prediction, salary_range):
        prediction = round(int(prediction), -2)
        min_salary = prediction - salary_range
        max_salary = prediction + salary_range
        return {"min": min_salary, "max": max_salary}

# Add a selectbox to the sidebar:
model_select = st.sidebar.selectbox(
    'Models',
    os.listdir("models/")
)

if model_select == "xgb_model":
    model = XGB.Booster()
    model.load_model(f"models/{model_select}")
else:
    model =  pickle.load(open(f"models/{model_select}", 'rb'))

# Add a slider to the sidebar:
salary_range = st.sidebar.slider(
    'Select a range of salary',
    0, 2000, (1000)
)

####################
# Page title
####################

st.write("""
    # Salary prediction app in IT
    This app predicts the **salary** values of IT job offert.  
""")

####################
# Set form values
####################

df = pd.read_csv("../data/production.csv")

cities = df['city'].value_counts().index
technologies = df['marker_icon'].value_counts().index
workplace_type = df['workplace_type'].value_counts().index
experience_level = df['experience_level'].value_counts().index
contract_type = df['contract_type'].value_counts().index
remote_interview = ['True', 'False']
remote = ['True', 'False']
company_size = ['micro', 'small', 'medium', 'large']

# ######################
# # Inputs
# ######################

col1, col2 = st.beta_columns(2)

city_input = col1.selectbox(
    'City',
    cities
)

workplace_type_input = col1.selectbox(
    'Workplace type',
    workplace_type
)

experience_level_input = col1.selectbox(
    'Experience level',
    experience_level
)

remote_interview_input = col2.radio(
    'Remote Interview',
    remote_interview
)

remote_input = col2.radio(
    'Remote',
    remote
)

company_size_input = st.selectbox(
    'Company size',
    company_size
)


technology_input = st.selectbox(
    'Technology',
    technologies
)

contract_type_input = st.radio(
    'Contract Type:',
    contract_type
)
