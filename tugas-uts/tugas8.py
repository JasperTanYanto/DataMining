from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

#implemetasi phyton2
iris = datasets.load_iris()
features = iris.data
plt.scatter(features[:,0],features[:,1])
plt.show()

#implemetasi phyton3
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
#implemetasi phyton4
from sklearn.metrics import silhouette_samples,silhouette_score
wcss=[]
for i in range(1, 11):
    kmeans=KMeans(n_clusters= i, init='k-means++',max_iter = 300,n_init=10,random_state= 0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number ofcluster')
    plt.ylabel('WCSS')
    plt.show()
#implement 5
kmeans=KMeans(n_clusters= i, init='k-means++',max_iter = 300,n_init=10,random_state= 0)
pred_y=kmeans.fit_predict(features)
plt.scatter(features[:,0],features[:,1])
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers[:,1],s=300,c='red')
plt.show()