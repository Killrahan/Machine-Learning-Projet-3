import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer

def fill(dataset):
    """Given a dataset and indexes of missing lines, returns a dataset with filled lines

    Args:
        dataset (array_like): whole dataset containing all the useful sensors
        indexes (array_like): indexes[f][2] are fully missing lines in file f, indexes[f][3] are partially missing in file f

    Returns:
        array_like: the dataset, but with 
    """

    for f in range(len(dataset)):
        impute = KNNImputer(n_neighbors=1, missing_values=-999999.99)
        dataset[f] = impute.fit_transform(dataset[f])
        
    
    return dataset