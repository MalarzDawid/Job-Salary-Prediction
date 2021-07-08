#!/usr/bin/env python
# coding: utf-8

# # Przygotowanie danych

# # Importowanie bibliotek

# In[1]:


# Import the necessary packages
import os
import requests
import pickle
import pandas as pd


# # Pobranie danych z serwisu justjoin.it

# In[2]:


request = requests.get("https://justjoin.it/api/offers")
data = request.json()

df = pd.DataFrame.from_dict(data)


# In[3]:


# Create dir and export csv file
os.makedirs("../../data", exist_ok=True)
df.to_csv("../../data/justjoin_jobs.csv")


# # Dane

# In[4]:


df.head()


# In[5]:


# Get info about columns type and non-null values
df.info()


# # Wartości Null

# In[6]:


# Check null values
df.isnull().sum()


# In[7]:


# Find null values in country_code column
df.loc[df["country_code"].isnull()]


# In[8]:


# Changing the values of the country_code from NaN to "PL"
df["country_code"] = df["country_code"].fillna("PL")


# # Dodawanie nowych cech

# ## Rodzaj umowy

# In[9]:


# Custom function
def get_contract_type(data):
    return data[0]["type"]


# In[10]:


df["contract_type"] = df["employment_types"].apply(get_contract_type)


# In[11]:


# Browsing unique contr"act types
df["contract_type"].unique()


# ## Wynagrodzenie (Target)

# In[12]:


def get_salary(data, label="from"):
    if data[0]["salary"] is not None:
        return data[0]["salary"][label]
    else:
        return data[0]["salary"]
    
def get_currency(data):
    if data[0]["salary"] is not None:
        return data[0]["salary"]["currency"]
    else:
        return None


# In[13]:


# Get salary min
df["salary_min"] = df["employment_types"].apply(get_salary)

# Get salary max
df["salary_max"] = df["employment_types"].apply(get_salary,label="to")

# Get currency
df["currency"] = df["employment_types"].apply(get_currency)


# ## Umiejętności i Technologie

# ### Technologie

# In[14]:


# Get number of technology per offert
df["num_technology"] = df["skills"].apply(len)


# In[15]:


# Custom function to get skills and level from dictionary

def get_skill(x):
    skills = []
    for num_skills in range(len(x)):
        try:
            skills.append(x[num_skills]["name"])
        except:
            continue
    return skills

def get_level(x):
    levels = []
    for num_skills in range(len(x)):
        try:
            levels.append(x[num_skills]["level"])
        except:
            continue
    return levels


# In[16]:


df["technology"] = df["skills"].apply(get_skill)
df["levels"] = df["skills"].apply(get_level)


# In[17]:


# Check our results
df[["technology", "levels"]]


# In[18]:


# Browsing unique marker_icon => technology
technologies = df["marker_icon"].unique()


# ## Koniec

# In[20]:


# Show our dataframe
df.head()


# ### Eksport danych do pliku CSV

# In[21]:


df.to_csv("../../data/justjoin_offers.csv", index=False)


# In[22]:


df.to_pickle("../../data/justjoin_offers.data")

