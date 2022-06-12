import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Data Analyse",
    page_icon="📈",
)

@st.cache(persist=True)
def fetch_and_clean_data():
    data = pd.read_csv("players.csv")

    data = data.drop_duplicates(subset=['PLAYER'])
    
    data = data[data['POS'] != 'Res']
    data = data[data['POS'] != 'Sub']
    
    data.describe(include = 'all')
    
    data = data.replace(['LWB', 'LB', 'RB', 'RWB', 'LCB', 'RCB'], 'CB')
    data = data.replace(['CDM', 'LM', 'RM', 'CAM', 'RCM', 'RDM', 'LCM', 'LDM', 'LAM', 'RAM'], 'CM')
    data = data.replace(['LW', 'LF', 'ST', 'RF', 'RW', 'LS', 'RS'], 'CF')
   
    return data


data = fetch_and_clean_data()

data.info()

data.describe(include = 'all')

data.POS.unique()

labels = data["POS"]

import seaborn as sns

st.title("I) Data structure")
st.dataframe(data.head())

st.title("II) Number of players by position")
valueCounts = labels.value_counts()
fig, ax = plt.subplots()
ax.pie(valueCounts.values, labels=valueCounts.index, autopct='%1.1f%%',
        shadow=True, startangle=90)

st.pyplot(fig)
st.bar_chart(labels.value_counts(), width=0, height=0, use_container_width=True)


st.title("II) Correlation between features")
fig, ax = plt.subplots()
sns.heatmap(data.corr(),annot=True,cmap="RdYlGn", ax=ax)
st.write(fig)

st.title("III) Outliers detection")

for i in data[["PAC", "SHO", "PAS", "DRI", "DEF", "PHY"]]:
    fig, ax = plt.subplots()
    sns.boxplot(data[i], ax=ax)
    st.write(fig)



