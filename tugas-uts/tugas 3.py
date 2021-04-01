# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import chardet

#import dataset
#strip value to float

dataset = pd.read_csv('DataBukub.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
print(X)
print(Y)    
#taking care mising data(nan)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,2:3])
X[:,2:3] = imputer.transform(X[:,2:3])
print(X)
#encodingkategori data
#encoding indepedent variabel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
#encoding dependet
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)
print(Y)
#Spliting 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.4,random_state = 1)
print(X_train)
print(Y_train)
print(X_test)
print(Y_test)                       
#feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,2:3] = sc.fit_transform(X_train[:,2:3])
X_test[:,2:3] = sc.transform(X_test[:,2:3])
print(X_train)
print(X_test)
