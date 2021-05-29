import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
iris=datasets.load_iris()
df=pd.DataFrame(iris['data'])
print(df.head())

#import the written fungsion
from scipy.cluster.vq import whiten
scaled_data = whiten(df.to_numpy())

#import cluster and linkage function
from scipy.cluster.hierarchy import fcluster,linkage
#use the linkage
distances_matrix=linkage(scaled_data,method='ward',metric='euclidean')
#import the dendrogam function
from scipy.cluster.hierarchy import dendrogram
#create dendrogram
dn=dendrogram(distances_matrix)
#display
plt.show()
