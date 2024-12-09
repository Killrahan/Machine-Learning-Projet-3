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


def get_subject_sensors(subject_array, subject_id):
    """
    Return indexex of time series associated to a particular subject. 
    """
    index = 0
    index_array = []
    for subject in subject_array:
        if subject == subject_id:
            index_array.append(index)

        index+=1

    return index_array




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


def build_dataset(useless_th, nb_subject, nb_tot, Z_th, method="average"):
    """
    This function get rid of useless sensors, fill missing time series using either an
    averaging method or KNN_imputation and build the sets X_train, y_train, X_validation
    and y_validation that will be used to assess the model. 

    Args : 
    useless_th : number of missing series require to toss a sensor data. 
    nb_subject : number of subjects that will be used to create the learning set. 
    nb_tot : number of subjects. 
    method : method to use to fill the fissing data (average or knn_imput)

    return : [X_train, y_train, X_validation, y_validation]
    """
    if not (method == "average" or method == "knn_imput"):
        raise Exception("Method shoud be average or knn_imput")

    if not isinstance(useless_th, int):
        raise Exception("Invalid Threshold.")

    if nb_subject > nb_tot:
        raise Exception(
            "Cannot pick more subjects than there is in the dataset.")

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

    #Getting indexes associated to subjects that are part of training and validation sets. 
    subject_array = np.loadtxt(os.path.join(LS_path, 'subject_id.txt'))
    training_indexes = np.array([])
    validation_indexes = np.array([])

    for i in range(1,nb_tot+1):
        curr_indexes = get_subject_sensors(subject_array,i)
        #print(f"Current indexes = {curr_indexes}")
        if i <= nb_subject:
            training_indexes = np.append(training_indexes,curr_indexes)
            #print(training_indexes)
        else:
            validation_indexes = np.append(validation_indexes,curr_indexes)
            #print(validation_indexes)
    
    training_indexes = training_indexes.ravel().tolist()
    validation_indexes = validation_indexes.ravel().tolist()
    training_indexes.sort()
    validation_indexes.sort()
    
    """print(f"Training indexes : {training_indexes} \n")
    print(f"Validation indexes : {validation_indexes} \n")
    print("e")"""

    # Build sets and fill missing data :

    X_train = np.zeros((len(training_indexes), (len(f_indexes)*512)))
    X_validation = np.zeros((len(validation_indexes), (len(f_indexes)*512)))
    X_test = np.zeros((3500, (len(f_indexes) * 512)))

    y = np.loadtxt(os.path.join(LS_path, 'activity_Id.txt'))
    y_train = []
    y_validation = []
    for j in training_indexes:
            y_train.append(y[int(j)])
    for j in validation_indexes:
            y_validation.append(y[int(j)])

    #print(f"y_train = {y_train} \n\n\n y_validation = {y_validation} \n\n\n")

    if method == "average":
        data_array = []
        for f in f_indexes:
            data_curr = np.loadtxt(os.path.join(
                LS_path, 'LS_sensor_{}.txt'.format(f)))
            Deal_Outliers.deal_outliers(data_curr, Z_th=Z_th)
            data_array.append(fill_average(
                data_curr, [missing_array[f-2][2], missing_array[f-2][3]]))

    if method == "knn_imput":
        data_array = []
        for f in f_indexes:
            data_curr = np.loadtxt(os.path.join(
                LS_path, 'LS_sensor_{}.txt'.format(f)))
            Deal_Outliers.deal_outliers(data_curr, Z_th=Z_th)
            data_array.append(data_curr)

        data_array = fill_knn(data_array)

    index = 0

    for f in f_indexes:
        k = 0 
        print(f"f = {f} \n")
        for line_index in training_indexes:
            X_train[:, (index)*512:(index+1) *
                        512][k] = data_array[index][int(line_index)]
            k+=1
        k = 0 
        for line_index in validation_indexes:
            X_validation[:, (index)*512:(index+1) *
                             512][k] = data_array[index][int(line_index)]
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
    my_set = build_dataset(3500,5,5,Z_th=np.inf,method="knn_imput")
    X_train = my_set[0]
    y_train = my_set[1]
    X_validation = my_set[2]
    y_validation = my_set[3]
    X_test = my_set[4]
        
    clf = RandomForestClassifier(n_estimators=1250,max_features=1)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    toy_script.write_submission(y_pred)

           
    """clf = RandomForestClassifier(n_estimators=1000,max_features=1)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    toy_script.write_submission(y_pred)"""

   
    # Tuning parameters of the random forest :
    """param_grid = [{'max_features': np.arange(1, 21), 'criterion': ['gini','entropy','log_loss']}]
    clf = GridSearchCV(RandomForestClassifier(n_estimators=100), param_grid)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    grid_predictions = clf.predict(X_validation)
    print(classification_report(y_validation, grid_predictions))
    """