#import library

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
dataset = pd.read_csv('konsumen.csv')
dataset.keys()
dataku=pd.DataFrame(dataset)
dataku.head()
X=np.asarray(dataset)
print(X)
#menampilkan  data dalam bentuk scatter plot
plt.scatter(X[:,0],X[:,1],label='TruePosition')
plt.xlabel("gaji")
plt.ylabel("pengeluaran")
plt.title("grafik penyebaran")
plt.show()
#mengaktifkan kmeans dengan jumlah ke 2
kmeans=KMeans(n_clusters=2)
kmeans.fit(X)
#menampilkan  nilai centroid
print(kmeans.cluster_centers_)
#plot Data point
#mevisualisasi hasil klaster
plt.scatter(X[:,0],X[:,1],c =kmeans.labels_,cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='black')
plt.xlabel("Gaji")
plt.ylabel("pengeluaran")
plt.title("grafikhasil klasterisasi data gaji")
plt.show()
