# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 13:21:46 2021

@author: basti
"""

import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from joblib import dump

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer

from sklearn.cluster import KMeans

df=pd.read_csv("../data/base_tennis_atp.csv",sep=";")

#### PREDICTIONS
def maj_model_tennis():
    df.loc[df['seed'].isna(),'seed'] = 100
    
    df_reg=df.copy()
    df_reg.loc[(df_reg['seed']<100),"classe"] = True
    df_reg.loc[(df_reg['seed']>=100),"classe"] = False
    
    df_reg = df_reg.dropna()
    
    df_reg.loc[(df_reg['result']=="win"),"win"] = 1
    df_reg.loc[(df_reg['result']=="lose"),"win"] = 0
    
    df_cluster = df_reg
    df_cluster = df_cluster[["name","ht","classe","age","ace_car","df_car","nb_set_car"]]
    df_cluster = df_cluster.set_index(['name'])
    
    x = df_cluster.values 
    
    kmeans = KMeans(n_clusters=20, random_state=0).fit(x)
    kmeans.labels_
    df_cluster['cluster'] = kmeans.labels_
    df_cluster.loc[df_cluster.cluster == 0].count()
    
    df_cluster = df_cluster.reset_index()
    
    df_cluster=df_cluster.rename(columns={"name":"advers"})
    df_cluster=df_cluster[["advers","age","cluster"]]
    df_reg=df_reg.merge(df_cluster,on=["advers","age"],how="outer")
    
    ### MODELE ACE
    count = df_reg.groupby('name').count().reset_index()
    count = count[['name',"tourney_id"]]
    count = count.rename(columns={"tourney_id":"count"})
    indexNames = count[ (count['count'] < 7.5)].index
    count.drop(indexNames , inplace=True)
    
    df_reg = pd.merge(df_reg, count, on=['name'])
    
    df_cluster.to_csv("../data/Cluster_tennis.csv",sep=";")
    
    df_ML=df_reg[["win","ace",'age',"tourney_id","nb_set",
            'ht',
            'name',
            'surface',
            'tourney_level',
            'classe',
            'minutes',
            "cluster",
            "advers"]]
    
    df_ML=df_ML.dropna()
    
    df_ML.to_csv("../data/tennis_atp-master/base_atp_ace.csv",sep=";")
    
    y = df_ML[['ace']]
    
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
    
    RFR = make_pipeline(preprocessor, RandomForestRegressor(n_estimators=20,max_depth=100,
                                                            min_samples_leaf=1))
    RFR.fit(X, y)
    
    dump(RFR,"../models/predi_ace")
    
    
    ### MODELE WIN
    
    df_w=df_reg[["tourney_id","win","ace",'age',
            'ht',
            'name',
            'surface',
            'tourney_level',
            'classe',
            'minutes',
            "cluster",
            "advers"]]
    
    count = df_w.groupby(['name',"advers"]).count().reset_index()
    count = count[['name',"advers","tourney_id"]]
    count = count.rename(columns={"tourney_id":"count"})
    indexNames = count[ (count['count'] < 7.5)].index
    count.drop(indexNames , inplace=True)
    
    df_w = pd.merge(df_w, count, on=['name',"advers"])
    
    df_w=df_w.dropna()
    
    df_w.to_csv("../data/tennis_atp-master/base_atp_win.csv",sep=";")
    
    b = df_w[['win']]
    
    A = df_w[['age',
            'ht',
            'name',
            'surface',
            'tourney_level',
            'classe',
            'minutes',
            "advers"
            ]]
    
    numeric_data_w = ['age',
                    'minutes',
                    'ht'
                    ]
    object_data_w = ['name',
                   'surface',
                   'tourney_level',
                   'classe',
                   "advers"
                   ]
    
    A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.1, random_state=2)
    
    numeric_pipeline = make_pipeline(PolynomialFeatures(2),StandardScaler())
    object_pipeline = make_pipeline(OneHotEncoder())
    
    preprocessor = make_column_transformer((numeric_pipeline, numeric_data_w),
                                           (object_pipeline, object_data_w))
    
    MLP = make_pipeline(preprocessor, MLPRegressor(solver="lbfgs",learning_rate="constant",
                                                   hidden_layer_sizes=(10,),alpha=0.001,
                                                   activation="tanh"))
    
    
    MLP.fit(A, b)
    
    dump(MLP,"../models/predi_win")
    
    ### MODELE NB SET
    d = df_ML[['nb_set']]
    
    C = df_ML[['age',
            'ht',
            'name',
            'surface',
            'tourney_level',
            'classe',
            'minutes',
            "cluster"
            ]]
    
    numeric_data_s = ['age',
                    'minutes',
                    'ht'
                    ]
    object_data_s = ['name',
                   'surface',
                   'tourney_level',
                   'classe',
                   "cluster"
                   ]
    
    #PIPELINE
    
    C_train, C_test, d_train, d_test = train_test_split(C, d, test_size=0.1, random_state=0)
    
    numeric_pipeline = make_pipeline(PolynomialFeatures(2),StandardScaler())
    object_pipeline = make_pipeline(OneHotEncoder())
    
    preprocessor = make_column_transformer((numeric_pipeline, numeric_data_s),
                                           (object_pipeline, object_data_s))
    
    SGD = make_pipeline(preprocessor, SGDRegressor())
    
    SGD.fit(C, d)
    
    dump(RFR,"../models/predi_set")



