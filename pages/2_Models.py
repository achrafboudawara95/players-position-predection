# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 11:40:21 2022

@author: nayma
"""

import pandas as pd
import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier as xgb
from sklearn.metrics import classification_report
import streamlit as st

st.set_page_config(
    page_title="Model selection",
    page_icon="🗼",
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

features = data[["PAC", "SHO", "PAS", "DRI", "DEF", "PHY"]]

labels = data["POS"]

le = LabelEncoder()
encodedLabels = le.fit_transform(labels)
encodedLabels = encodedLabels.reshape(-1,1)

def getMetrics(modelName, yTest, model):
    metrics = classification_report(yTest, model, output_dict=True)
    avg = metrics.get("weighted avg")
    return [modelName, metrics.get("accuracy"), avg.get("f1-score"), avg.get("precision"), avg.get("recall")]

xTrain, xTest, yTrain, yTest = train_test_split(features, encodedLabels, test_size=0.33, random_state=0)

modelxgb = xgb(booster='gbtree',
            objective='multi:softprob', max_depth=5,
            learning_rate=0.1, n_estimators=100)

modelxgb.fit(xTrain, yTrain)

predxgb = modelxgb.predict(xTest)

xgbMetrics = getMetrics("xgboost", yTest, predxgb)

from sklearn.neighbors import KNeighborsClassifier
modelknn = KNeighborsClassifier(n_neighbors=10)

modelknn.fit(xTrain,yTrain)
predknn= modelknn.predict(xTest)

knnMetrics = getMetrics("knn", yTest, predknn)

from sklearn.ensemble import RandomForestClassifier

modelRforest = RandomForestClassifier(random_state=0,n_estimators=100,max_depth=10)

modelRforest.fit(xTrain, yTrain)

predRforest = modelRforest.predict(xTest)

rForestMetrics = getMetrics("random forest", yTest, predRforest)

from sklearn import svm

modelSVC = svm.SVC(kernel='poly')
    
modelSVC.fit(xTrain, yTrain)
    
predSVC = modelSVC.predict(xTest)

svmMetrics = getMetrics("svm", yTest, predSVC)

df = pd.DataFrame(
    [xgbMetrics, knnMetrics, rForestMetrics, svmMetrics],
    columns=["model", "accuracy", "f1-score", "precision", "recall"])

st.title("I) Models performence")
st.table(df.style.highlight_max(axis=0))

@st.cache(persist=True, suppress_st_warning=True)
def gridSearch():
    #################################################################
    # Pipeline
    #################################################################
    params = {
        'booster': 'gbtree',
        'max_depth': 5,
        'objective': 'multi:softprob',
        'n_estimators':100
    }
    pipe_xgb = Pipeline([
        ('clf', xgb(**params))
        ])
    
    parameters_xgb = {
            'clf__n_estimators':[100,200], 
            'clf__max_depth':[5,10], 
        }
    
    grid_xgb = GridSearchCV(pipe_xgb,
        param_grid=parameters_xgb,
        scoring='accuracy',
        cv=5,
        refit=True)
    
    #################################################################
    # Modeling
    #################################################################
    start_time = time.time()
    
    grid_xgb.fit(xTrain, yTrain)
    
    #Calculate the score once and use when needed
    acc = grid_xgb.score(xTest,yTest)
    
    st.title("II) Grid Search")
    
    st.text("Best params                        : %s" % grid_xgb.best_params_)
    st.text("Best training data accuracy        : %s" % grid_xgb.best_score_)    
    st.text("Best validation data accuracy (*)  : %s" % acc)
    st.text("Modeling time                      : %s" % time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    
gridSearch()