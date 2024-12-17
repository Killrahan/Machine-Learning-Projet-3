import numpy as np

from sklearn.ensemble import RandomForestClassifier



if __name__ == "__main__":
    dataset = build_dataset(3500,5,5,13,method="knn_imput")
    X_train = dataset[0]
    y_train = dataset[1]
    X_test = dataset[4]
    clf = RandomForestClassifier(n_estimators=1500,n_jobs=-1)
