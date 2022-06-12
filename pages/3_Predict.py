import pickle
import pandas as pd
from xgboost import XGBClassifier as xgb
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Predict position",
    page_icon="🤨",
)

modelxgb = pickle.load(open('D:/achraf/tek-up/DSN2/ProjetII/finalized_model.sav', 'rb'))

col1, col2 = st.columns(2)
PAC = col1.slider('Pace rating', 0, 100, 0)
SHO = col2.slider('Shooting rating', 0, 100, 0)
col1, col2 = st.columns(2)
PAS = col1.slider('Passing rating', 0, 100, 0)
DRI = col2.slider('Dribbling rating', 0, 100, 0)
col1, col2 = st.columns(2)
DEF = col1.slider('Defensive rating', 0, 100, 0)
PHY = col2.slider('Physicality rating', 0, 100, 0)

labels = ['CF', 'CM', 'CB', 'GK']
le = LabelEncoder()
encodedLabels = le.fit_transform(labels)
encodedLabels = encodedLabels.reshape(-1,1)

if st.button('Predict'):
    row = pd.DataFrame([[PAC,SHO,PAS,DRI,DEF,PHY]], columns=["PAC", "SHO", "PAS", "DRI", "DEF", "PHY"])
    pos = modelxgb.predict(row)
    st.write('Position : %s'% le.inverse_transform(pos)[0])