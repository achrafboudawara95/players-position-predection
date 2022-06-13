import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from xgboost import XGBClassifier as xgb

data = pd.read_csv("players.csv")

data = data.drop_duplicates(subset=['PLAYER'])

data.info()

data.describe(include = 'all')

data = data[data['POS'] != 'Res']
data = data[data['POS'] != 'Sub']

data.describe(include = 'all')

data = data.replace(['LWB', 'LB', 'RB', 'RWB', 'LCB', 'RCB'], 'CB')
data = data.replace(['CDM', 'LM', 'RM', 'CAM', 'RCM', 'RDM', 'LCM', 'LDM', 'LAM', 'RAM'], 'CM')
data = data.replace(['LW', 'LF', 'ST', 'RF', 'RW', 'LS', 'RS'], 'CF')



data.POS.unique()

features = data[["PAC", "SHO", "PAS", "DRI", "DEF", "PHY"]]

labels = data["POS"]

le = LabelEncoder()
encodedLabels = le.fit_transform(labels)
encodedLabels = encodedLabels.reshape(-1,1)

xTrain, xTest, yTrain, yTest = train_test_split(features, encodedLabels, test_size=0.33, random_state=0)

modelxgb = xgb(booster='gbtree',
            objective='multi:softprob', max_depth=5,
            learning_rate=0.1, n_estimators=100)

modelxgb.fit(xTrain, yTrain)

# save the model to disk
import pickle
filename = 'finalized_model.sav'
pickle.dump(modelxgb, open(filename, 'wb'))