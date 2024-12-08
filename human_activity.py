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
import Deal_Outliers
"""
This file is the file where we built machine learning's method to predict the
activity. 
"""

# The following function will be used to know which sensor shouldn't be considered.


def number_missing():
    """
    Return the number and indexes (corresponding to a particular line) of time series 
    that are either fully or partially missing for each sensors. 
    Return value takes this shape : [[fully_missing2,partially_missing2,fully_missing_indexes2, partially_missing_indexes2]
                                    ,...,[fully_missing32,partially_missing32,fully_missing_indexes32, partially_missing_indexes32]]

    """
    FEATURES = range(2, 33)
    LS_path = os.path.join('./', 'LS')
    missing_array = []
    for f in FEATURES:
        fully_index = 0
        partially_index = 0
        partially_missing = 0
        fully_missing = 0
        fully_indexes = []
        partially_indexes = []
        data = np.loadtxt(os.path.join(LS_path, 'LS_sensor_{}.txt'.format(f)))
        for line in data:
            missing_element = 0
            for element in line:
                if element == -999999.99:
                    missing_element += 1
            if missing_element == len(line):
                fully_missing += 1
                fully_indexes.append(fully_index)
            elif missing_element != 0:
                partially_missing += 1
                partially_indexes.append(partially_index)
            fully_index += 1
            partially_index += 1

        missing_array.append(
            [fully_missing, partially_missing, fully_indexes, partially_indexes])

    return missing_array


def fill_average(dataset, indexes):
    """
    Fill the missing values in a sensor dataset.
    args : 
    dataset : the dataset to be modified. (1 txt file).
    indexes : the indexes of line of time series that are either partially missing or
              fully_missing. indexes[0] contains all time series that are fully missing
              and indexes[1] contains all time series that are partially missing.
    method: defines which method to use.  average or knn_imput
    return : a modified dataset.
    """
    dataset_mean = []

    for i in range(len(dataset)):
        if not (-999999.99 in dataset[i]):
            dataset_mean.append(np.average(dataset[i]))

    dataset_mean = np.average(dataset_mean)
    # print("Mean of the dataset : ", dataset_mean)

    for i in range(len(dataset)):
        # Deal with fully missing time series.
        if i in indexes[0]:
            for j in range(len(dataset[i])):
                dataset[i][j] = dataset_mean

        elif i in indexes[1]:
            """if (i == 944):
                print(f"index table : {indexes[1]} \n")
                print(f"dataset line : {dataset[i]} \n")"""

            non_missing = []

            for j in range(len(dataset[i])):
                if dataset[i][j] != -999999.99:
                    non_missing.append(dataset[i][j])

            """if len(non_missing) < 1:
                print(f"empty non_missing for i = {i}\n")"""

            average = np.average(non_missing)
            # print(f"average = {average}")

            for j in range(len(dataset[i])):
                if dataset[i][j] == -999999.99:
                    dataset[i][j] = average

    return dataset


def build_dataset(useless_th, nb_series, nb_tot, Z_th, method="average", Random=False):
    """
    This function get rid of useless sensors, fill missing time series using either an
    averaging method or KNN_imputation and build the sets X_train, y_train, X_validation
    and y_validation that will be used to assess the model. 

    Args : 
    useless_th : number of missing series require to toss a sensor data. 
    nb_series : Number of time_series we want to put in the training set.
    nb_tot : number of time series in the original dataset. 
    method : method to use to fill the fissing data (average or knn_imput)

    return : [X_train, y_train, X_validation, y_validation]
    """
    if not (method == "average" or method == "knn_imput"):
        raise Exception("Method shoud be average or knn_imput")

    if not isinstance(useless_th, int):
        raise Exception("Invalid Threshold.")

    if nb_series > nb_tot:
        raise Exception(
            "Cannot pick more series than there is in the dataset.")

    LS_path = os.path.join('./', 'LS')
    TS_path = os.path.join('./', 'TS')
    # Select the sensor we keep : (NB : for the moment only average method is built).

    missing_array = number_missing()
    # print(f"missing_array = {missing_array}")
    f_indexes = []
    f_index = 2
    for i in missing_array:
        if i[0] < useless_th:
            f_indexes.append(f_index)
        f_index += 1
    f_index = 0
    # print(f"f_indexes = {f_indexes}")

    # Build sets and fill missing data :
    X_train = np.zeros((nb_series, (len(f_indexes)*512)))
    X_validation = np.zeros((nb_tot - nb_series, (len(f_indexes)*512)))
    X_test = np.zeros((nb_tot, (len(f_indexes) * 512)))

    if method == "average":
        data_array = []
        for f in f_indexes:
            data_curr = np.loadtxt(os.path.join(
                LS_path, 'LS_sensor_{}.txt'.format(f)))
            Deal_Outliers.deal_outliers(data_curr, Z_th=5)
            data_array.append(fill_average(
                data_curr, [missing_array[f-2][2], missing_array[f-2][3]]))

    if method == "knn_imput":
        data_array = []
        for f in f_indexes:
            data_curr = np.loadtxt(os.path.join(
                LS_path, 'LS_sensor_{}.txt'.format(f)))
            Deal_Outliers.deal_outliers(data_curr, Z_th=1.3)
            data_array.append(data_curr)

        data_array = fill_knn(data_array)

    index = 0
    if not Random:
        y_train = np.loadtxt(os.path.join(
            LS_path, 'activity_Id.txt'))[:nb_series]
        y_validation = np.loadtxt(os.path.join(
            LS_path, 'activity_Id.txt'))[nb_series:]
        for f in f_indexes:
            print(f"f = {f} \n")
            X_train[:, (index)*512:(index+1) *
                    512] = data_array[index][:nb_series]
            X_validation[:, (index)*512:(index+1) *
                         512] = data_array[index][nb_series:]
            data = np.loadtxt(os.path.join(
                TS_path, 'TS_sensor_{}.txt'.format(f)))
            X_test[:, (index)*512:(index+1)*512] = data
            index += 1

    if Random:
        y = np.loadtxt(os.path.join(LS_path, 'activity_Id.txt'))
        train_line = np.random.choice(3500, nb_series, replace=False)
        train_line = np.sort(train_line)
        #print(f"Train line : {train_line} \n")
        validation_line = []
        y_train = []
        y_validation = []
        #print(f"Validation line : {validation_line}")
        for j in range(3500):
            if not (j in train_line):
                validation_line.append(j)
                y_validation.append(y[j])
            else:
                y_train.append(y[j])
        """print(f"y_train = {y_train} \n")
        print(f"y_validation = {y_validation}")
        print(f"Validation line : {validation_line}")"""

        """At this point train_line contains nb_series line indexes and 
        validation_line nb_tot - nb_series lines indexes."""

        for f in f_indexes:
            k = 0
            print(f"f = {f} \n")
            for line_index in train_line:
                X_train[:, (index)*512:(index+1) *
                        512][k] = data_array[index][line_index]

                k += 1
            k = 0
            for line_index in validation_line:
                X_validation[:, (index)*512:(index+1) *
                             512][k] = data_array[index][line_index]
                k += 1
            data = np.loadtxt(os.path.join(
                TS_path, 'TS_sensor_{}.txt'.format(f)))
            X_test[:, (index)*512:(index+1)*512] = data
            index += 1

    """print('X_train size: {}.'.format(np.shape(X_train)))
    print('y_train size: {}.'.format(np.shape(y_train)))
    print('X_validation size: {}.'.format(np.shape(X_validation)))
    print('y_validation size: {}.'.format(np.shape(y_validation)))
    print('X_test size : {}.'.format(np.shape(X_test)))
    print(f"X_train = {X_train}")"""
    return X_train, y_train, X_validation, y_validation, X_test


if __name__ == '__main__':
    print("Main file.")
    # Below parameters will be tuned : 
    """optimal_array = []
    #Tuning useless_th: 
    useless_th_tune = [100,500,3500]
    optimum_th = 100
    optimum_score = 0
    score_av = 0

    for j in useless_th_tune:
        my_set = build_dataset(j, 1167, 3500, Z_th=2, method="knn_imput", Random=True)
        X_train = my_set[0]
        y_train = my_set[1]
        X_validation = my_set[2]
        y_validation = my_set[3]
        X_test = my_set[4]
        score_array = []
        for i in range(10):
            clf = RandomForestClassifier(n_estimators=100, max_features=4)
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_validation)
            score = accuracy_score(y_predict, y_validation)
            score_array.append(score)
            print(f"score = {score} for useless_th = {j}")
        
        score_av = np.mean(score_array)
        print(f"Average score = {score_av} for useless_th = {j}")
        if score_av > optimum_score:
            optimum_score = score_av 
            optimum_th = j 

    optimal_array.append(optimum_th)
    print("Optimal parameters : \n", optimal_array)

    #tuning Z_th: 
    Z_th_tune = np.arange(10,21,1)/10 
    optimum_score = 0
    optimum_Z_th = 1
    score_av = 0
    for j in Z_th_tune:
        my_set = build_dataset(optimal_array[0], 1167, 3500, Z_th=j, method="knn_imput", Random=True)
        X_train = my_set[0]
        y_train = my_set[1]
        X_validation = my_set[2]
        y_validation = my_set[3]
        X_test = my_set[4]
        score_array = []
        for i in range(10):    
            clf = RandomForestClassifier(n_estimators=100, max_features=4)
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_validation)
            score = accuracy_score(y_predict, y_validation)
            score_array.append(score)
            print(f"score = {score} for Z_th = {j}")

        score_av = np.mean(score_array)
        print(f"Average score = {score_av} for Z_th = {j}")
        if score_av > optimum_score:
            optimum_score = score_av 
            optimum_Z_th = j 

    optimal_array.append(optimum_Z_th)
    print("Optimal parameters : \n", optimal_array)

    #Tuning max_features : 
    my_set = build_dataset(optimal_array[0], 1167, 3500, Z_th=optimal_array[1], method="knn_imput", Random=True)
    X_train = my_set[0]
    y_train = my_set[1]
    X_validation = my_set[2]
    y_validation = my_set[3]
    X_test = my_set[4]

    max_features_tune = np.arange(1,51)
    optimum_score = 0
    optimum_max = 1
    score_av = 0

    for j in max_features_tune:
        print("j = ",j)
        score_array = []
        for i in range(10):
            clf = RandomForestClassifier(n_estimators=100, max_features=j)
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_validation)
            score = accuracy_score(y_predict, y_validation)
            print(f"score = {score} for max_features = {j}")
            score_array.append(score)
        
        score_av = np.mean(score_array)
        print(f"Average score = {score_av} for max_features = {j}")
        if score_av > optimum_score:
            optimum_score = score_av
            optimum_max = j 
    
    optimal_array.append(optimum_max)
    print("Optimal parameters : \n", optimal_array)"""
    """
    Up to now, the following optimal parameters have been found : 
    useless_th = 3500 (don't exclude any sensors)
    Z_th = 1.7 (further tests proved that dealing with outliers actually reduce the accuracy...)
    max_features = 4 

    Mean score with these parameters (Z_th -> inf) : 95,5 % on validation set. 
    it remains to analyze other parameters of random forest or to use another algorithm. 
    
    """
    #Write submission. 
    my_set = build_dataset(3500, 1167, 3500, Z_th=20000000, method="knn_imput", Random=True)
    X_train = my_set[0]
    y_train = my_set[1]
    X_validation = my_set[2]
    y_validation = my_set[3]
    X_test = my_set[4]
    array = []

    for j in range(2,52):
        clf = RandomForestClassifier(max_features=4, min_samples_split=j)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_validation)
        score = accuracy_score(y_predict,y_validation)
        print(f"score for min_leaf = {j} : {score}")
        array.append(score)
        #toy_script.write_submission(y_predict)
    score_av = np.mean(array)
    print("average = ", score_av)

    array = []
    for j in range(500):
        clf = RandomForestClassifier(max_features=4, min_weight_fraction_leaf=j/100)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_validation)
        score = accuracy_score(y_predict,y_validation)
        print(f"score for min_leaf = {j} : {score}")
        array.append(score)
        #toy_script.write_submission(y_predict)
    score_av = np.mean(array)
    print("average = ", score_av)
    aray = []
    for j in range(500):
        clf = RandomForestClassifier(max_features=4, min_impurity_decrease=j/1000)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_validation)
        score = accuracy_score(y_predict,y_validation)
        print(f"score for min_leaf = {j} : {score}")
        array.append(score)
        #toy_script.write_submission(y_predict)
    score_av = np.mean(array)
    print("average = ", score_av)