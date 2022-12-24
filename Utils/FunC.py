from sklearn.decomposition import KernelPCA
import numpy as np

def Dimensionality_reduction(data,n_components):
    return KernelPCA(n_components=n_components,kernel='linear').fit_transform(data)

def normalize(data, max, min, type):
    m,n=data.shape
    output=np.ones((m,n))
    if type==0:
        for i in range(m):
            output[i,:]=(data[i,:]-min)/(max-min)
    if type==1:
        min_matrix=min*np.ones((m,n))
        for i in range(m):
            output[i,:]=(max-min)*data[i,:]+min_matrix[i,:]
    return output