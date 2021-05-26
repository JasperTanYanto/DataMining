import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import time
import itertools


#baca dataku
datasetPath = "Dataku.txt"
dataset = np.loadtxt(datasetPath, delimiter =" ")
#mendefinisikan prameter k-means klustering
k=2#jumlah kluster yang diinginkan
iterationCounter=0#counter untuk iterasi
input=dataset#inpuit data
#fungsi untuk inisialisasi titik pusat kluster
def initCentroid(dataIn,k):
    result=dataIn[np.random.choice(dataIn.shape[0],k,replace=False)]
    return result
#fungsi unutk plot hasil klaster
def plotClusterResult(listClusterMembers,centroid,iteration,converged):
    n=listClusterMembers._len_( )
    color=iter(cm.rainbow(np.linspace(0,1,n)))
    plt.figure("result")
    plt.clf()
    plt.title("iteration-"+iteration)
    marker=itertools.cycle(('.','*', '^','x','+'))
    for i in range(n):
        col=next(color)
        memberCluster=np.asmatrix(listClusterMembers[i])
        plt.scatter(np.ravel(memberCluster[:,0]), np.ravel(memberCluster[:,1]),marker=marker._next_(),
                    c=col,label="centroid-"+str(i+1) )
        if(converged==0):
            plt.legend()
            plt.ion()
            plt.show()
            plt.pause(0.1)
        if(converged ==1):  
            plt.legend()
            plt.show(block=True)
            
#fungsi utama algoritma k_means
def kMeans(data,centroidInit):
    nCluster=k#banyaknya klaster
    global iterationCounter
    centroidInit=np.matrix(centroidInit)
    #lopping konvergen
    while(True):
        iterationCounter +=1
        euclideanMatrixAllCluster=np.ndarray(shape=(data.shape[0],0))
        #ulangi proses
        for i in range(0,nCluster):
            centroidRepeated = np.repeat(centroidInit[i,:],data.shape[0],axis=0)
            deltaMatrix = abs(np.subtract(data,centroidRepeated ))
            #hitung jarak euclidean
            euclideanMatrix=np.sqrt(np.square(deltaMatrix).sum(axis=1)    )
            euclideanMatrixAllCluster=\
                np.concatenate((euclideanMatrixAllCluster,euclideanMatrix),axis=1)
        #tempatkan data keklaster terdekat
        clusterMatrix = np.ravel(np.argmin(np.matrix(euclideanMatrixAllCluster),axis=1))
        listClusterMembers=[[]for i in range(k)]
        for i in range (0, data.shape[0]):#assihg data cluster to cluster regarding
            listClusterMembers[np.asscalar(clusterMatrix[i])].append(data[i,:])

        #hitung titik pusat
        newCentroid=np.ndarray(shape=(0,centroidInit.shape [1]))
        for i in range (0,nCluster):
            memberCluster = np.asmatrix(listClusterMembers[i])
            centroidCluster = memberCluster.mean(axis=0)
            newCentroid = np.concatenate((newCentroid ,centroidCluster),axis=0)
            print("iter:",iterationCounter)
            print("centroid:",newCentroid)

        #break dari loop jika sudah konvergen
        if((centroidInit==newCentroid).all()):
            break
        #update titik pusat klaster dng baru
        centroidInit=newCentroid
        #plot hasil klaster literasi
        plotClusterResult(listClusterMembers,centroidInit,str(iterationCounter),0)
        time.sleep(1)#diberi jeda 1 detik  
        return listClusterMembers,centroidInit
        #panggil fungsi inisial klaster
        centroidInit=initCentroid(input,k)
        #panggil fuungsi kmeans
        clusterResult,centroid=kMeans(input,centroidInit)
        #plot hasil final klaster
        plotClusterResult(clusterResult,centroid,str(iterationCounter)+"(converged",1)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
            
            
            
            
            
            
            