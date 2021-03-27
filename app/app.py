####################
# Imports libraries
####################
import pickle
import pandas as pd
import streamlit as st
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

####################
# Custom functions
####################
def create_salary_range(prediction):
        salary_range = 1000
        prediction = round(int(prediction), -2)
        min_salary = prediction - salary_range
        max_salary = prediction + salary_range
        return {"min": min_salary, "max": max_salary}

####################
# Load model and dataframe
####################
with open("models/linear_model", "rb") as file:
    model = pickle.load(file)

df = pd.read_csv("../data/model_df.csv")

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

cities = df['city'].value_counts().index[:10]
technologies = df['marker_icon'].value_counts().index
workplace_type = df['workplace_type'].value_counts().index
experience_level = df['experience_level'].value_counts().index
contract_type = df['contract_type'].value_counts().index
remote_interview = ['True', 'False']
remote = ['True', 'False']
company_size = ['very_small', 'small', 'medium', 'large']

######################
# Inputs
######################

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
    company_size,
    help="""
very_small (0-30)  
small (30-100)  
medium (100-1000)  
large (1000-10000+)  
"""
)


technology_input = st.selectbox(
    'Technology',
    technologies
)

contract_type_input = st.radio(
    'Contract Type:',
    contract_type
)

######################
# Pre-built model
######################

if st.button('Predict!'):
    X = df.drop('salary_mean', axis=1)
    user_inputs = [city_input, technology_input, workplace_type_input, experience_level_input, contract_type_input, company_size_input]

    user_inputs_df = pd.DataFrame([user_inputs], columns=X.columns)
    X = X.append(user_inputs_df)

    # Feature engineering
    X['city'] = X['city'].astype('category')
    X['city'] = X['city'].cat.codes
    X['marker_icon'] = X['marker_icon'].astype('category')
    X['marker_icon'] = X['marker_icon'].cat.codes
    # Get dummies
    X = pd.get_dummies(X, columns=['workplace_type', 'experience_level', 'contract_type', 'company_size_bin'], prefix="feature")

    # Get user inputs
    predict_value = X.tail(1)
    X = X[:-1]

    # Prediction
    prediction = model.predict(predict_value)[0]
    salary= create_salary_range(prediction)

    # Print prediction salary range
    st.success(f"Prediction salary: {salary['min']} - {salary['max']}")