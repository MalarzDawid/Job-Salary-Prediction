#!/usr/bin/env python
# coding: utf-8

# # Predykcja - Uczenie Maszynowe

# # Importowanie bibliotek

# In[1]:


import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from catboost import Pool

# Models
import xgboost as XGB
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


# # Dane

# In[2]:


df = pd.read_csv("../../data/full_df.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


to_drop = [
    "title",
    "street",
    "address_text",
    "company_name",
    "company_url",
    "company_size",
    "latitude",
    "longitude",
    "published_at",
    "id",
    "employment_types",
    "company_logo_url",
    "skills",
    "salary_min",
    "salary_max",
    "currency",
    "num_technology",
    "technology",
    "levels",
    "remote"
]


# In[6]:


print("Before: ", len(df))
df = df.drop(to_drop, axis=1)
df = df.dropna()
print("After: ", len(df))


# In[7]:


df.to_csv("../../data/production.csv", index=False)


# # Własne funkcje

# In[8]:


def save_model(name, model):
    with open(f"../../app/models/{name}", "wb") as file:
        pickle.dump(model, file)


# # Przygotowanie danych

# ## Label Encoder

# In[10]:


cols = ["workplace_type", "country_code", "contract_type", "company_size_bin"]
df[cols] = df[cols].apply(LabelEncoder().fit_transform)
df.head()


# ## GetDummies

# In[11]:


features_bool = df.select_dtypes(include="bool").columns
df = pd.get_dummies(df, columns=features_bool, drop_first=True)

features_object = df.select_dtypes(include="object").columns
df = pd.get_dummies(df, columns=features_object)


# In[12]:


df.head()


# # Tworzenie modeli

# In[13]:


results = pd.DataFrame(columns=["model", "mae"])


# In[14]:


X = df.drop(["salary_mean"], axis=1)
y = df["salary_mean"]


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.3, random_state=2021
)
print(X_train.shape)
print(X_test.shape)


# In[16]:


# model evaluation function

def model_evaluate(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


# ## RandomForestRegressor

# In[18]:


forest_regressor_model = RandomForestRegressor(
    n_estimators=300, max_depth=7, random_state=2021
).fit(X_train, y_train)

cv_model = cross_val_score(
    forest_regressor_model, X_train, y_train, cv=10, scoring="neg_mean_absolute_error"
)
y_preds = forest_regressor_model.predict(X_test)
cv_model_mean = np.mean(cv_model)

print("Cross val score:", cv_model_mean)
print("MAE: ", model_evaluate(y_test, y_preds))

results = results.append(
    {"model": "forest_regressor_model", "mae": model_evaluate(y_test, y_preds)},
    ignore_index=True,
)
save_model("forest_regressor_model", forest_regressor_model)


# ## GradientBoostingRegressor

# In[20]:


gradient_boosting_model = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.01, max_depth=6, random_state=2021
).fit(X_train, y_train)

cv_model = cross_val_score(
    gradient_boosting_model, X_train, y_train, cv=10, scoring="neg_mean_absolute_error"
)
y_preds = gradient_boosting_model.predict(X_test)
cv_model_mean = np.mean(cv_model)

print("Cross val score:", cv_model_mean)
print("MAE: ", model_evaluate(y_test, y_preds))

results = results.append(
    {"model": "gradient_boosting_model", "mae": model_evaluate(y_test, y_preds)},
    ignore_index=True,
)
save_model("gradient_boosting_model", gradient_boosting_model)


# ## Xgboost

# In[22]:


xgb_model = XGB.XGBRegressor(
    base_score=0.5,
    booster="gbtree",
    learning_rate=0.1,
    max_depth=4,
    n_estimators=180,
    random_state=2021,
).fit(X_train, y_train)

cv_model = cross_val_score(
    xgb_model, X_train, y_train, cv=10, scoring="neg_mean_absolute_error"
)

y_preds = xgb_model.predict(X_test)
cv_model_mean = np.mean(cv_model)

print("Cross val score:", cv_model_mean)
print("MAE: ", model_evaluate(y_test, y_preds))

results = results.append(
    {"model": "xgb_model", "mae": model_evaluate(y_test, y_preds)}, ignore_index=True
)
xgb_model.save_model("../../app/models/xgb_model")


# ## Catboost

# In[24]:


train_dataset = Pool(X_train, y_train)
test_dataset = Pool(X_test, y_test)

catboost_model = CatBoostRegressor(
    iterations=300, learning_rate=0.1, depth=6, silent=True, random_state=2021
).fit(train_dataset)

cv_model = cross_val_score(
    catboost_model, X_train, y_train, cv=10, scoring="neg_mean_absolute_error"
)

y_preds = catboost_model.predict(X_test)
cv_model_mean = np.mean(cv_model)

print("Cross val score:", cv_model_mean)
print("MAE: ", model_evaluate(y_test, y_preds))

results = results.append(
    {"model": "catboost_model", "mae": model_evaluate(y_test, y_preds)},
    ignore_index=True,
)
save_model("catboost_model", catboost_model)


# ## Wyniki

# In[25]:


results.head()


# In[26]:


results.to_csv("../../data/training_results.csv")

