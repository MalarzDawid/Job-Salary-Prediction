#!/usr/bin/env python
# coding: utf-8

# # Analiza Danych

# Głównym założeniem projektu, jest stworzenie systemu, który na podstawie danych dotyczących zatrudnienia takich jak *miasto*, *technologia*, *tryb pracy*, *umowa zatrudnia* itp. będzie przewidywał jakie wynagrodzenie powinno znaleźć się w danej ofercie pracy. Jednak, zanim przejdę do budowania takiego systemu wykorzystując uczenie maszynowe, pozwolę sobie zrobić analizę danych, które pobrałem z nieoficjalnego API strony **justjoin.it**. 

# ## Importowanie bibliotek

# In[1]:


import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

get_ipython().run_line_magic('matplotlib', 'inline')

# Config
sns.set_theme(style="darkgrid")
sns.set()


# ## Dane

# Dane, które wykorzystuję w tym notebooku, przeszły już wcześniej małą obrókę. 
# 
# Dodałem kilka kolumn takich jak:  
#    - contract_type
#    - salary_min
#    - salary_max
#    - num_technology
#    - technology
#    - level
# 
# Posłużą one w późniejszej części na stworzenie dodatkowych cech, a także ułatwią, niektóre z moich analiz.
#   

# In[2]:


df = pd.read_pickle("../data/justjoin_offers.data")


# In[3]:


df.head()


# In[4]:


df.info()


# W większości dane, są typu **object**, które przechowują tekst, jednak mamy również **float64** w którym kryje się nasz główny *target*, jakim jest płaca minimalna oraz maksymalna. Można również zauważyć, że w niektórych ofertach brakuje informacji o pensji, co lepiej pokaże poniższa wizualizacja

# In[5]:


fig = px.imshow(df.isnull(), color_continuous_scale="Viridis")
fig.show()


# # Analiza danych

# Zanim zaczniemy budować nasz pierwszy model, do przewidywania wynagrodzenia, warto spojrzeć na dane. Na samym początku zaczniemy od **targetu**, czyli zmiennej, którą będziemy chcieli przewidywać.
# 
# Strona justjoin.it operuje na widełkach wynagrodzenia, jeżeli przyjmiemy, że naszym **targetem**, będzie `salary_min`, czyli minimalna kwota wynagrodzenia, to predykcje mogą być niedoszacowane. Jeżeli weźmiemy `salary_max` to wynagrodzenia będą za duże. 
# 
# W ten o to sposób, dochodzimy do momentu, kiedy trzeba stworzyć specjalną zmienną, która stanie się naszym **targetem**. Moim pierwszym pomysłem i aktualnym jest stworzenie średniej arytmetycznej z `salary_min` oraz `salary_max` w ten sposób, będziemy przewidywać wynagrodzenie, które znajduje się w widełkach. 

# In[6]:


df["salary_mean"] = df[["salary_min", "salary_max"]].mean(axis=1)


# Nie zapominajmy o jednej istotnej kwestii. Firmy, które zamieszczają oferty pracy, płacą w różnych walutach. My chcemy, aby nasz target miał wynagrodzenie w złotówkach, dlatego zamieńmy inne waluty na złotówki, według stałego kursu. 

# In[7]:


df["currency"].value_counts()


# In[8]:


currency = {"usd": 3.90, "eur": 4.60, "gbp": 5.40, "chf": 4.20, "pln": 1.0}


# Target stworzony - waluty zamienione na złotówki. Czas na wizualizacje, które odpowiedzą nam na kilka podstawowych pytań.

# In[9]:


fig = px.histogram(
    df,
    x="salary_mean",
    opacity=0.8,
    title="Salary mean",
    marginal="box",
)
fig.show()


# Widzimy, że dwie oferty wyglądają bardzo ciekawie. Jedna oferuje **62.5k** złotych, a druga ok. **75k**. Sprawdźmy, co to za firmy i kogo poszukują

# In[10]:


df.loc[df["salary_mean"] > 60000]


# Oho... Nasz rekordzista to oferta pracy we Wrocławiu jako `C++ Group Interview`, natomiast mamy też akcent z naszego kraju `Director of Engineering (REMOTE)`. Zweryfikujmy jeszcze dane, tak, aby mieć pewność, że nie ma w nich żadnej pomyłki.

# Jeżeli **DevsData** oraz **Consult Red** tak dużo płacą za specjalistę w naszym kraju, to sprawdźmy też jak wyglądają ich inne oferty zatrudnienia. (Weźmy pierwszą 10)

# In[11]:


df.loc[df["company_name"] == "Consult Red"][
    ["title", "marker_icon", "contract_type", "salary_mean"]
].head(10)


# In[12]:


df.loc[df["company_name"] == "DevsData LLC"][
    ["title", "marker_icon", "contract_type", "salary_mean"]
].head(10)


# Sprawdźmy jeszcze, jak poszczególne zmienne korelują z naszym **targetem**.
# 
# Na razie mamy mało danych numerycznych, ale już teraz możemy zauwazyć, że możliwość pracy zdalnej ma wpływ na wynagordzenie. 

# In[13]:


plt.figure(figsize=(15, 7))
df_corr = df.corr()

sns.heatmap(data=df_corr, annot=True, linewidths=2);


# ## Miasta

# Przejdźmy teraz do miast, które również odgrywają istotną rolę w kwocie jaką będziemy zarabiać u naszego teraźniejszego/przyszłego pracodwacy.
# 
# Na samym początku sprawdźmy w jakich miastach jest najwięcej ofert pracy

# In[14]:


# Get first ten cities from column 'city'
cities = df["city"].value_counts()[:10]

fig = px.bar(
    cities,
    x=cities.index,
    y=cities.values,
    title="Top 10 miast z największą ilością ofert",
    labels={"y": "Liczba ofert", "index": "Miasto"},
)
fig.show()


# Warszawa - ciekawe 🤔
# 
# Tak naprawdę nic ciekawego w tym nie ma, że Warszawa znajduje się na pierwszym miejscu, w końcu to stolica naszego kraju. 
# 
# Zobaczmy teraz, jak to się ma do zarobków, w TOP 10 miastach 

# In[15]:


fig = go.Figure()
for city in cities.index:
    mean = df.loc[df["city"] == city]["salary_mean"]
    fig.add_trace(go.Box(y=mean, name=city))
fig.show()


# ## Kraje

# W większości oferty pracy znajdujące się na justjoin.it pochodzą z naszego kraju, dlatego ciekawiej będzie sprawdzić z jakich krajów pochodzą pozostałe oferty.

# In[16]:


# Top 5 countries
countries = df["country_code"].value_counts()[1:6]  # First element = Poland

fig = px.bar(
    countries,
    x=countries.index,
    y=countries.values,
    labels={"y": "Liczba ofert", "index": "Kraj"},
    title="TOP 5 - Kraje z największą ilością ofert pracy (poza Polską)",
)
fig.show()


# Na samym początku projektu, przyjąłem założenie, że model, który docelowo ma działać, będzie funkcjonował tylko dla ofert w Polsce, dlatego też usuńmy wszystkie pozostałe kraje.

# In[17]:


print("Before:", len(df))
df = df[df["country_code"] == "PL"]
print("After:", len(df))


# Na tym etapie warto będzie jeszzce wrócić do miast i sprawdzić, czy żadne miasto nie jest w napisane w języku angielskim

# In[18]:


df["city"].unique()


# **Warsaw**, **Poland** bardzo ciekawie. Warsaw możemy zamienić na **Warszawę**, natomiast wiersz z Poland usuniemy.

# In[19]:


df["city"] = df["city"].replace("Warsaw", "Warszawa")
df["city"] = df["city"].replace("Варшава", "Warszawa")


# In[20]:


df = df[df["city"] != "Poland"]


# ## Technologie

# Justjoin.it oferuje piękną listę z różnymi technologiami do wyboru, sprawdźmy zatem jak prezentuje się obecnie rozkład technologi na krajowym rynku IT. 

# In[21]:


technologies = df["marker_icon"].value_counts()

fig = px.pie(
    technologies,
    values=technologies.values,
    names=technologies.index,
    title="Technologies",
    labels={"index": "Technology"},
)

fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=16)
fig.show()


# Ze wszystkich **4711** ofert, które posiadam na chwilę obecną, aż `17.6%` stanowią oferty związane z javascript. Niesamowite jest, jak ten język zdominował polski rynek, ale również zagraniczny... 
# 
# Python, który jest często polecany początkującym programistą, pomimo swoich wielu zastosowań, a także zainteresowania stanowi tylko `5.24%` wszystkich ofert w kraju.

# Wiemy już, jak rozkładają się punkty procentowe dla poszczególnych technologii. 
# 
# Szukjąc ofert pracy na justjoin.it mamy również możliwość sprawdzenia jakie biblioteki, technologie, frameworki są porządane przez naszego przyszłego pracodawcę, dlatego dobrym pomysłem będzie sprawdzenie, co oprócz samego poznania języka, da nam możliwość znalezienia pracy. 
# 
# Na początek stwórzmy główny dataframe, w którym będziemy trzymali wszystkie technologie wraz z bibliotekami, których się używa. Gdy już uda nam się to zrobić, wtedy tak przygotowane dane wyczyścimy ze zbędnych informacji, aby zminimalizować ilość zbędnych frameworków.   
# (Tak wiem `python 2.7` różni się od `python 3` itp.) 

# In[22]:


# String array => array
#df["technology"] = df["technology"].apply(literal_eval)


# In[23]:


# Create another dataframe
technologies = df[["marker_icon", "technology"]]


# In[24]:


all_technology = []

for marker_icon, technology_list in zip(
    technologies["marker_icon"], technologies["technology"]
):
    for technology in technology_list:
        all_technology.append([marker_icon, technology])


# In[25]:


tech_df = pd.DataFrame(all_technology, columns=["marker_icon", "tech"])


# In[26]:


tech_df.head()


# In[27]:


tech_df["tech"] = tech_df["tech"].str.lower()


# In[28]:


# Remove marker_icon from tech column
tech_df = tech_df[tech_df["marker_icon"] != tech_df["tech"]]


# In[29]:


tech_df["tech"] = tech_df["tech"].str.replace(r"[0-9]+", "")
tech_df["tech"] = tech_df["tech"].str.replace("php .x", "php")
tech_df["tech"] = tech_df["tech"].str.replace(".js", "")
tech_df["tech"] = tech_df["tech"].str.replace("github", "git")
tech_df["tech"] = tech_df["tech"].str.replace(r'[-+<>,./ \s]', "")
tech_df["tech"] = tech_df["tech"].str.replace("js", "javascript")

tech_df["tech"] = tech_df["tech"].str.replace("reac", "react")
tech_df["tech"] = tech_df["tech"].str.replace("reactt", "react")
tech_df["tech"] = tech_df["tech"].str.replace("angielski", "english")

tech_df["tech"] = tech_df["tech"].str.replace("uxdesign", "ux")
tech_df["tech"] = tech_df["tech"].str.replace("uidesign", "ui")


# In[30]:


# Delete rare technology
tech_df = tech_df.groupby(tech_df.columns.tolist(), as_index=False).size()
to_drop = tech_df[tech_df["size"] <= 10].index
tech_df.drop(to_drop, inplace=True)


# Wygląda na to, że mamy już wszystko co nam potrzebne, aby zobaczyć finalny efekt :)

# In[31]:


# FINAL
fig = px.sunburst(tech_df, path=["marker_icon", "tech"], values="size", height=800)
fig.show()


# ## Rodzaj pracy

# Covid dużo zmienił w branży. Dwa lata temu ten wykres kołowy wyglądałby zdecydowanie inaczej, zakładam, że praca w biurze stanowiłaby ponad `50%` wszystkich ofert.
# 
# Czasy się zmieniają i dane też, dlatego zobaczymy jak teraz prezentują się rodzaje pracy `zdalnie`, `hybrydowo`, `stacjonarnie`.

# In[32]:


df_workplace = df["workplace_type"].value_counts()

fig = go.Figure(
    data=[
        go.Pie(labels=df_workplace.index, values=df_workplace.values, pull=[0.1, 0, 0])
    ]
)
fig.show()


# # Wielkość firm

# Czy duże firmy płacą więcej niż małe startupy? 
# 
# Czy duże firmy zatrudniają więcej juniorów niż firmy składające się z 10 osób? 
# 
# Na te i inne pytania odpowiemy w tej części, tylko przygotujmy dane... 

# In[33]:


# Company size view
df["company_size"].head()


# In[34]:


df["company_size"] = df["company_size"].apply(lambda x: x.split("-")[0])


# In[35]:


df["company_size"] = df["company_size"].str.replace(r"\D", "")


# In[36]:


df = df.loc[df["company_size"] != ""]


# In[37]:


df["company_size"] = df["company_size"].apply(int)


# Sprawdźmy najpierw jak prezentuje się `boxplot` wielkości firm. Gdy już będziemy mieli jakieś wyobrażenie o tych danych, postaramy się je sensownie pogrupować. 

# In[38]:


fig = px.box(df, x="company_size")
fig.show()


# Mamy tak naprawdę tylko `1` firmy, które zatrudniają powyżej 200k pracowników. No właśnie... Firma zatrudniająca powyżej `200 000` pracowników... Sprawdziłem to i faktycznie, ktoś musiał się pomylić, ponieważ to raczej mało prawdopodobne - usuniemy te dane. Tutaj też warto zastanowić się, czy warto zostawiać firmy, których wielkości firm są większe niż np. `20 000`. Nie ma ich dużo, dlatego je również usuniemy

# In[39]:


df_big_company = df[df["company_size"] >= 200000]
df = df[df["company_size"] <= 20000]


# Sprawdźmy jeszcze na szybko z jakich firm pochodziły te `magiczne` oferty pracy

# In[40]:


df_big_company


# In[41]:


fig = px.box(df, x="company_size")
fig.show()


# In[42]:


df["company_size_bin"] = pd.cut(
    df["company_size"], bins=[0, 10, 50, 250, 200000], labels=["micro", "small", "medium", "large"]
)


# In[43]:


company_size = df["company_size_bin"].value_counts()

fig = go.Figure(
    data=[go.Bar(x=company_size.index, y=company_size.values)]
)

fig.update_layout(
    title="Wielkość firmy",
    yaxis=dict(title="Ilość firm"),
    xaxis=dict(title="Rodzaj firmy"),
)

fig.show()


# # Poziom doświadczenia

# Poziom doświadczenia będzie tak naprawdę kluczowy dla naszego późniejszego modelu, ponieważ to właśnie doświadczenie definiuje ile pracodawca będzie wstanie zapłacić za nowego pracownika. Nikogo nie zdziwi, że `junior`, który w branży dopiero stawia pierwsze kroki, będzie zarabiał kilkakrotnie mniej niż specjalista z 20 letnim doświadczeniem. Spradźmy na początku, kogo rynek pracy IT poszukuje teraz najczęściej, a także porównajmy płace dla każdego szczebla kariery.

# In[44]:


fig = px.histogram(df, x="experience_level")

fig.update_layout(
    title="Poziom doświadczenia",
    yaxis=dict(title="Liczba ofert"),
    xaxis=dict(title="Doświadczenie"),
)

fig.show()


# Ponad **2500** firm poszukuje pracowników na stanowisko mid. Dlaczego? Odpowiedź może być bardzo prosta, mid ma doświadczenie w branży, pracował przy min. 1 projekcie komerycjnym. Z punktu widzenia osoby zatrudniającej, osoba już z kilkuletnim doświadczeniem, będzie zdecydowanie lepszą inwestycją, niż osoba, która dopiero zaczyna swoją karierę. 
# 
# Sprawdźmy jeszcze jak prezentują się dane na wykresie kołowym w procentach. 

# In[45]:


fig = go.Figure(
    data=[
        go.Pie(
            labels=df["experience_level"],
            title="Poziom doświadczenia",
        )
    ]
)

fig.show()


# Tutaj warto zwrócić uwagę na jeszcze jedną rzecz. Gdy zaczęła się pandemia z wielu portali poznikały oferty na juniorów wynika to z faktu, że lepiej uczyć nową osobę stacjonarnie, niż rozwiązywać problemy na odległość. Na szczęście, obecnie styuacja już trochę się unormowała i ofert przybywa. 

# In[46]:


exp_mean = df.groupby(by=["experience_level"])["salary_mean"].mean()
fig = px.histogram(
    exp_mean,
    exp_mean.index,
    exp_mean.values,
    title="Średnie wynagrodzenie dla poszczególnych poziomów doświadczenia",
)

fig.update_layout(yaxis=dict(title="Pensja"), xaxis=dict(title="Doświadczenie"))

fig.show()


# # Eksport danych

# In[47]:


# Export dataframe with 29 columns
df.to_csv("../data/full_df.csv", index=False)

