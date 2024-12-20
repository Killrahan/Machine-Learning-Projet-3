import numpy as np
import os

from Resolve import get_subject_sensors
from Resolve import number_missing

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt


def set_stat(line):
    array = [np.mean(line), np.var(line), np.min(line), np.max(line), np.max(line) - np.min(line), np.percentile(line, 25), np.percentile(line, 50),
             np.percentile(line, 75), np.percentile(line, 100)]
    fft = np.fft.fft(line)

    for i in fft:
        array.append(np.linalg.norm(i))

    return array


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


def build_dataset(useless_th, nb_tot):
    """
    This function get rid of useless sensors, fill missing time series using simple imputer
     method, scale the data using robust scaler and build the sets X, y and X_test.

    Args : 
    useless_th : number of missing series require to toss a sensor data.  
    nb_tot : Total number of subjects. 

    return : [X_train, y_train, X_validation, y_validation]
    """

    LS_path = os.path.join('./', 'LS')
    TS_path = os.path.join('./', 'TS')

    # equivalent to the for loop in Resolve.py with all sensors kept
    # f_indexes = np.arange(2,33)
    missing_array = number_missing()
    f_indexes = []
    f_index = 2
    for i in missing_array:
        if i[0] < useless_th:
            f_indexes.append(f_index)
        f_index += 1
    f_index = 0

    print(f"Kept sensors : {f_indexes}")

    # Getting indexes associated to subjects ids
    subject_array = np.loadtxt(os.path.join(LS_path, 'subject_Id.txt'))

    # create indexes as a list of empty np arrays
    subject_indexes = []

    # subject_indexes[i] contains the indexes of subject_id = i
    for i in range(1, nb_tot+1):
        curr_indexes = get_subject_sensors(subject_array, i)
        subject_indexes.append(curr_indexes)

    # Build sets and fill missing data :

    X = []
    for i in range(nb_tot):
        X.append(np.zeros((len(subject_indexes[i]), (len(f_indexes)*521))))

    X_test = np.zeros((3500, (len(f_indexes) * 521)))

    y_data = np.loadtxt(os.path.join(LS_path, 'activity_Id.txt'))
    y = []

    # y[i] contains all the elements of activity_id where subject_id = i
    for i in range(nb_tot):
        y_i = []
        for j in subject_indexes[i]:
            y_i.append(y_data[int(j)])
        y.append(y_i)

    # load data from files and preprocess it
    data_array = []
    data_test_array = []
    for f in f_indexes:
        data_curr = np.loadtxt(os.path.join(
            LS_path, 'LS_sensor_{}.txt'.format(f)))
        data_curr_test = np.loadtxt(os.path.join(
            TS_path, 'TS_sensor_{}.txt'.format(f)))
        data_array.append(data_curr)
        data_test_array.append(data_curr_test)

    data_array = fill(data_array)

    # data_array = fill_knn(data_array)

    for i in range(len(data_array)):
        transformer = RobustScaler().fit(data_array[i])
        data_curr = transformer.transform(data_array[i])
        transformer = RobustScaler().fit(data_test_array[i])
        data_curr = transformer.transform(data_test_array[i])

    for i in range(nb_tot):
        index = 0
        for f in f_indexes:
            k = 0
            print(f"f = {f} \n")
            for line_index in subject_indexes[i]:
                X[i][:, (index)*521:(index+1) *
                     521][k] = set_stat(data_array[index][int(line_index)])
                k += 1

            for line in range(3500):
                X_test[:, (index)*521:(index+1) *
                       521][line] = set_stat(data_test_array[index][line])

            index += 1

    return X, y, X_test


def build_subsets(X, y, validation_index, split=True):

    X_train = []
    y_train = []
    X_validation = []
    y_validation = []

    if split:
        X_validation = np.array(X[validation_index])
        y_validation = np.array(y[validation_index])

        for i in range(len(X)):
            if i != validation_index:
                X_train.append(X[i])
                y_train.append(y[i])

        X_train = np.concatenate([np.array(x) for x in X_train])
        y_train = np.concatenate([np.array(y_i) for y_i in y_train])

    else:
        X_train = X
        y_train = y

    return X_train, y_train, X_validation, y_validation


def score(X, y, i, n_estimators, max_depth, max_features):

    X_train, y_train, X_validation, y_validation = build_subsets(X, y, i)
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
    clf.fit(X_train, y_train)

    return clf.score(X_validation, y_validation)


def CV_score(dataset, n_estimators, max_depth, max_features):

    X = dataset[0]
    y = dataset[1]
    X_test = dataset[2]

    K = len(X)

    scores = np.zeros(K)

    for i in range(K):
        print(i)
        score_i = score(X, y, i, n_estimators, max_depth, max_features)
        scores[i] = score_i
        print(score_i)

    return np.average(scores)


if __name__ == "__main__":

    average_array = []
    missing_th_array = [100, 200, 500, 3500]
    for i in missing_th_array:
        dataset = build_dataset(i, 5)
        score_Av = CV_score(dataset, 100, None, 'sqrt')
        print(f"Average = {score_Av} for missing_th = {i}")
        average_array.append(score_Av)

    plt.plot(missing_th_array, average_array)
    plt.xlabel(r"$missing_{th}$")
    plt.ylabel(r"Score")
    plt.ylim(0, 1)
    plt.xlim(0, 3800)
    plt.show()
