import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

"""

The name of this file is not well chosen. It should be called "imputation" as it is the 
file we call in other script to fill the missing data.


"""


def fill(dataset):
    """Given a dataset, returns the dataset with filled lines

    Args:
        dataset (array_like): whole dataset containing all the useful sensors

    Returns:
        array_like: the filled dataset
    """

    for f in range(len(dataset)):
        impute = SimpleImputer(missing_values=-999999.99)
        dataset[f] = impute.fit_transform(dataset[f])

    return dataset
