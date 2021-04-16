# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import tree

#membaca dataset dari frame
irisDataset = pd.read_csv('kdi.csv',delimiter=',',header=0)

#mengubah class kolom('species)dari string ke unique-integer
irisDataset["Species"] = pd.factorize(irisDataset.Species)[0]

#menghapus kolom id
irisDataset = irisDataset.drop(labels="Id", axis=1)

#mengubah dataframe ke array numpy
#irisDataset=irisDataset.as_matrix()
irisDataset = irisDataset.to_numpy()

#mebagi dataset,40 baris data untuk training
#dan 20 baris data untuk testing
dataTraining = np.concatenate((irisDataset[0:40,:],irisDataset[50:90,:]),axis=0)
dataTesting = np.concatenate((irisDataset[40:50,:],irisDataset[90:100,:]),axis=0)

#memcah dataset keinput
inputTraining = dataTraining[:,0:4]
inputTesting = dataTraining[:,0:4]
labelTraining = dataTraining[:,4]
labelTesting = dataTraining[:,4]

#mendefinisikan decision tree clasifier
model = tree.DecisionTreeClassifier()

#mentrining model
model = model.fit(inputTraining,labelTraining)

#memprediksi input data ntesting
hasilPrediksi = model.predict(inputTesting)
print("label sebenarnya",labelTesting)
print("hasil sebenarnya:",hasilPrediksi)

#menghitung akurasi
prediksiBenar = (hasilPrediksi == labelTesting).sum()
prediksiSalah = (hasilPrediksi != labelTesting).sum()
print("prediksi benar",prediksiBenar,"data")
print("prediksi salah",prediksiSalah,"data")
print("akurasi:",prediksiBenar/(prediksiBenar+prediksiSalah)*100,"%")
