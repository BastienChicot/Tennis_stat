# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 13:11:17 2021

@author: basti
"""

import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import make_column_transformer

from sklearn.cluster import KMeans

df=pd.read_csv("Data/base_tennis.csv",sep=";")

#### PREDICTIONS
df.loc[df['seed'].isna(),'seed'] = 100

import statsmodels.formula.api as smf

df_reg=df.copy()
df_reg.info()
df_reg.loc[(df_reg['seed']<100),"classe"] = True
df_reg.loc[(df_reg['seed']>=100),"classe"] = False

df_reg = df_reg.dropna()
df_reg.info()
reg_lin = smf.ols('ace ~ np.log(age) + np.log(ht) + C(surface) + C(tourney_level) + C(classe) +'
                 ' C(hand) + np.log(minutes) + df ' ,
                  data=df_reg).fit()

reg_lin.summary()

df_reg.loc[(df_reg['result']=="win"),"win"] = 1
df_reg.loc[(df_reg['result']=="lose"),"win"] = 0

df_cluster = df_reg
df_cluster = df_cluster[["name","ht","classe","age","ace_car","df_car","nb_set_car"]]
df_cluster = df_cluster.set_index(['name'])

x = df_cluster.values #returns a numpy array
 
#Choix du nombre de cluster
inertie = []
K_range = range(1, 50)
for k in K_range:
    model = KMeans(n_clusters=k).fit(x)
    inertie.append(model.inertia_)

plt.plot(K_range, inertie)
    
kmeans = KMeans(n_clusters=20, random_state=0).fit(x)
kmeans.labels_
df_cluster['cluster'] = kmeans.labels_
df_cluster.loc[df_cluster.cluster == 0].count()

df_cluster = df_cluster.reset_index()

df_cluster=df_cluster.rename(columns={"name":"advers"})
df_cluster=df_cluster[["advers","age","cluster"]]
df_reg=df_reg.merge(df_cluster,on=["advers","age"],how="outer")


count = df_reg.groupby('name').count().reset_index()
count = count[['name',"tourney_id"]]
count = count.rename(columns={"tourney_id":"count"})
indexNames = count[ (count['count'] < 10)].index
count.drop(indexNames , inplace=True)


##MODELE ACE
df_reg = pd.merge(df_reg, count, on='name')

df_cluster.to_csv("Data/Cluster_tennis.csv",sep=";")

df_ML=df_reg[["win","ace",'age',
        'ht',
        'name',
        'surface',
        'tourney_level',
        'classe',
        'minutes',
        "cluster"]]

df_ML=df_ML.dropna()

y = df_ML[['ace']]

#min_max_scaler = MinMaxScaler()
#y_scaled = min_max_scaler.fit_transform(y)
#y = pd.DataFrame(y_scaled)
#y = y[0]

X = df_ML[['age',
        'ht',
        'name',
        'surface',
        'tourney_level',
        'classe',
        'minutes',
        "cluster"
        ]]

numeric_data = ['age',
                'minutes',
                'ht'
                ]
object_data = ['name',
               'surface',
               'tourney_level',
               'classe',
               "cluster"
               ]

#PIPELINE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

numeric_pipeline = make_pipeline(PolynomialFeatures(2),StandardScaler())
object_pipeline = make_pipeline(OneHotEncoder())

preprocessor = make_column_transformer((numeric_pipeline, numeric_data),
                                       (object_pipeline, object_data))

#MODELE

from sklearn.ensemble import RandomForestRegressor

RFR = make_pipeline(preprocessor, RandomForestRegressor(n_estimators=20,max_depth=100,
                                                        min_samples_leaf=1))

from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluation (model):
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test.values.ravel(), y_pred)
    print(mse)
    print(mae)
    print (model.score(X_test, y_test))
    
evaluation(RFR)    

RFR.fit(X, y)
RFR.score(X,y)

from joblib import dump

dump(RFR,"predi_ace")
##MODELE WIN

df_ML=df_ML.dropna()

y = df_ML[['win']]

#min_max_scaler = MinMaxScaler()
#y_scaled = min_max_scaler.fit_transform(y)
#y = pd.DataFrame(y_scaled)
#y = y[0]

X = df_ML[['age',
        'ht',
        'name',
        'surface',
        'tourney_level',
        'classe',
        'minutes',
        "cluster"
        ]]

numeric_data = ['age',
                'minutes',
                'ht'
                ]
object_data = ['name',
               'surface',
               'tourney_level',
               'classe',
               "cluster"
               ]

#PIPELINE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

numeric_pipeline = make_pipeline(PolynomialFeatures(2),StandardScaler())
object_pipeline = make_pipeline(OneHotEncoder())

preprocessor = make_column_transformer((numeric_pipeline, numeric_data),
                                       (object_pipeline, object_data))

#MODELE

from sklearn.ensemble import RandomForestRegressor

RFR = make_pipeline(preprocessor, RandomForestRegressor(n_estimators=200,max_depth=200,
                                                        min_samples_leaf=1))

evaluation(RFR)    

RFR.fit(X,y)
RFR.score(X,y)

dump(RFR,"predi_win")

