import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import toy_script
import os
from knn_imputation import fill as fill_knn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import Deal_Outliers
import human_activity as HA 
from sklearn.ensemble import GradientBoostingClassifier


if __name__ == "__main__":

    optimum_score = 0
    optimum_max_features = 1
    score_av = 0
    my_set = HA.build_dataset(3500, 3, 5, Z_th=np.inf, method="knn_imput")
    X_train = my_set[0]
    y_train = my_set[1]
    X_validation = my_set[2]
    y_validation = my_set[3]
    X_test = my_set[4]
    score_array = []
    for i in range(100):    
        for j in range(1,101):
            clf = RandomForestClassifier(n_estimators=100, max_features=1,criterion="gini", max_depth=j)
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_validation)
            score = accuracy_score(y_predict, y_validation)
            score_array.append(score)
            print(f"score = {score} for max_features = {i}")

        score_av = np.mean(score_array)
        print(f"Average score = {score_av} for Z_th = {i}")
    

    """
    
    Conclusion : max_features = 1 est le meilleur. 
    
    """
    