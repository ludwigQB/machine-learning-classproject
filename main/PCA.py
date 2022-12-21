from sklearn.decomposition import KernelPCA


def Dimensionality_reduction(data,n_components):
    return KernelPCA(n_components=n_components,kernel='linear').fit_transform(data)

