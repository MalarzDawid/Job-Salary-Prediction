import requests
import pandas as pd

request = requests.get("https://justjoin.it/api/offers")
data = request.json()

df = pd.DataFrame.from_dict(data)
df.to_csv("justjoin_jobs.csv")
