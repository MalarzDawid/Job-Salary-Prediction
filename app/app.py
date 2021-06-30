# ####################
# # Imports libraries
# ####################
import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import xgboost as XGB
from sklearn.preprocessing import LabelEncoder

# Results
results_mae = pd.read_csv("../data/training_results.csv")

# Sidebar: selectbox
model_select = st.sidebar.selectbox("Models", os.listdir("models/"))

####################
# Custom functions
####################
def create_salary_range(prediction):
    salary_range = results_mae.loc[results_mae["model"] == model_select, "mae"].values[0]
    salary_range = round(int(salary_range), -2)
    prediction = round(int(prediction), -2)
    min_salary = prediction - salary_range
    max_salary = prediction + salary_range
    return {"min": min_salary, "max": max_salary}

# Sidebar: slider
# salary_range = st.sidebar.slider("Select a range of salary", 0, 2000, (1000))

# Sidebar: text
st.sidebar.write("## About Project: ")
st.sidebar.warning("My first idea about this project was create a website which predict salary"
                + "based on simple information e.g.city, technology, contract type." + 
                "This idea was successfully created. The next step is to improve my results and add someone features 🚀🚀🚀")
st.sidebar.code("Email: kontakt@malarzdawid.pl")

if model_select == "xgb_model":
    model = XGB.Booster()
    model.load_model(f"models/{model_select}")
else:
    model = pickle.load(open(f"models/{model_select}", "rb"))

####################
# Page title
####################

st.write("# Salary prediction app in IT")

####################
# Set form values
####################

df = pd.read_csv("../data/production.csv")

cities = df["city"].unique()
cities = np.sort(cities)
technologies = df["marker_icon"].value_counts().index
workplace_type = df["workplace_type"].value_counts().index
experience_level = df["experience_level"].value_counts().index
contract_type = df["contract_type"].value_counts().index
remote_interview = [True, False]
remote = [True, False]
company_size = ["micro", "small", "medium", "large"]

# ######################
# # Inputs
# ######################

col1, col2 = st.beta_columns(2)

city_input = col1.selectbox("City", cities)

workplace_type_input = col1.selectbox("Workplace type", workplace_type)

experience_level_input = col1.selectbox("Experience level", experience_level)

remote_interview_input = col2.radio("Remote Interview", remote_interview)

company_size_input = st.selectbox("Company size", company_size)

technology_input = st.selectbox("Technology", technologies)

contract_type_input = st.radio("Contract Type:", contract_type)

if st.button("Predict"):
    columns = {
        "city": city_input,
        "country_code": "PL",
        "marker_icon": technology_input,
        "workplace_type": workplace_type_input,
        "experience_level": experience_level_input,
        "remote_interview": remote_interview_input,
        "contract_type": contract_type_input,
        "salary_mean": 0,
        "company_size_bin": company_size_input,
    }
    df = df.append(columns, ignore_index=True)

    cols = ["workplace_type", "country_code", "contract_type", "company_size_bin"]
    df[cols] = df[cols].apply(LabelEncoder().fit_transform)

    features_bool = df.select_dtypes(include="bool").columns
    df = pd.get_dummies(df, columns=features_bool, drop_first=True)

    features_object = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=features_object)

    df = df.drop(["salary_mean"], axis=1)
    predict = df.tail(1)

    if model_select == "xgb_model":
        xgb_model = XGB.Booster()
        xgb_model.load_model("models/xgb_model")
        predict = XGB.DMatrix(predict)
        predicted_value = xgb_model.predict(predict)
    else:
        predicted_value = model.predict(predict)
    salary_range = create_salary_range(predicted_value)
    st.success(f"Salary: {salary_range['min']} - {salary_range['max']}")
