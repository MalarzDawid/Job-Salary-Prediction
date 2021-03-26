import pickle
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


with open("models/linear_model", "rb") as file:
    model = pickle.load(file)

df = pd.read_csv("../data/clear_output_data.csv")

st.header("""
    Create job offert with good salary!
""")

# Technology
cities = df['city'].value_counts().index[:10]
technologies = df['marker_icon'].value_counts().index
workplace_type = df['workplace_type'].value_counts().index
experience_level = df['experience_level'].value_counts().index
contract_type = df['contract_type'].value_counts().index
remote_interview = ['True', 'False']
remote = ['True', 'False']
company_size = ['very_small', 'small', 'medium', 'large']


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

# Contract type
contract_type_input = st.radio(
    'Contract Type:',
    contract_type
)

if st.button('Predict!'):
    X = df.drop('salary_mean', axis=1)
    
    inputs = [city_input, technology_input, workplace_type_input, experience_level_input, contract_type_input, company_size_input]
    
    df1 = pd.DataFrame([inputs], columns=X.columns)
    X = X.append(df1)

    X['city'] = X['city'].astype('category')
    X['city'] = X['city'].cat.codes

    X['marker_icon'] = X['marker_icon'].astype('category')
    X['marker_icon'] = X['marker_icon'].cat.codes
    X = pd.get_dummies(X, columns=['workplace_type', 'experience_level', 'contract_type', 'company_size_bin'], prefix="feature")

    predict_value = X.tail(1)
    X = X[:-1]

    predict = round(int(model.predict(predict_value)[0]), -2)
    average = 1000
    min_value = predict - average
    max_value = predict + average
    st.header("Done...")
    st.success(f"Salary: {min_value} - {max_value}")
