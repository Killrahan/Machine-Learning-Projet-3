import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import toy_script
import os
from knn_imputation import fill as fill_knn
from sklearn.metrics import pairwise_distances
from tensorflow.keras import layers, models

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


def build_dataset(useless_th, nb_series, nb_tot):
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

    if not isinstance(useless_th, int):
        raise Exception("Invalid Threshold.")

    if nb_series > nb_tot:
        raise Exception(
            "Cannot pick more series than there is in the dataset.")

    LS_path = os.path.join('./', 'LS')
    TS_path = os.path.join('./', 'TS')
    # Select the sensor we keep : (NB : for the moment only average method is built).

    missing_array = number_missing()
    f_indexes = []
    f_index = 2
    for i in missing_array:
        if i[0] < useless_th:
            f_indexes.append(f_index)
        f_index += 1
    f_index = 0

    # Build sets and fill missing data :
    X_train = np.zeros((nb_series, len(f_indexes),512))
    X_validation = np.zeros((nb_tot - nb_series, len(f_indexes),512))
    X_test = np.zeros((nb_tot, len(f_indexes) , 512))

    data_array = []
    for f in f_indexes:
        data_curr = np.loadtxt(os.path.join(
            LS_path, 'LS_sensor_{}.txt'.format(f)))

        data_array.append(data_curr)

    data_array = fill_knn(data_array)

    index = 0

    y = np.loadtxt(os.path.join(LS_path, 'activity_Id.txt'))
    train_line = np.random.choice(3500, nb_series, replace=False)
    train_line = np.sort(train_line)
    print(f"Train line : {train_line} \n")
    validation_line = []
    y_train = []
    y_validation = []
    print(f"Validation line : {validation_line}")
    for j in range(3500):
        if not (j in train_line):
            validation_line.append(j)
            y_validation.append(y[j])
        else:
            y_train.append(y[j])
    print(f"y_train = {y_train} \n")
    print(f"y_validation = {y_validation}")
    print(f"Validation line : {validation_line}")

    """At this point train_line contains nb_series line indexes and 
    validation_line nb_tot - nb_series lines indexes."""

    for f in f_indexes:
        k = 0
        print(f"f = {f} \n")
        for line_index in train_line:
            X_train[k][index] = data_array[index][line_index]

            k += 1
        k = 0
        for line_index in validation_line:
            X_validation[k][index] = data_array[index][line_index]
            k += 1
        data = np.loadtxt(os.path.join(TS_path, 'TS_sensor_{}.txt'.format(f)))
        for line_index in range(nb_tot):
            X_test[line_index][index] = data[line_index]
        index += 1
        
    X_train = X_train.transpose((0,2,1))
    X_validation = X_validation.transpose((0,2,1))
    X_test = X_test.transpose((0,2,1))

    print('X_train size: {}.'.format(np.shape(X_train)))
    print('y_train size: {}.'.format(np.shape(y_train)))
    print('X_validation size: {}.'.format(np.shape(X_validation)))
    print('y_validation size: {}.'.format(np.shape(y_validation)))
    print('X_test size : {}.'.format(np.shape(X_test)))
    print(f"X_train = {X_train}")
    return X_train, y_train, X_validation, y_validation, X_test


def create_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

if __name__ == '__main__':

    n_learn = int(0.2*3500)
    print("Main file.")
    LS_path = os.path.join('./', 'LS')
    my_set = build_dataset(3500, n_learn, 3500)
    X_train = my_set[0]
    y_train = my_set[1]
    X_validation = my_set[2]
    y_validation = my_set[3]
    X_test = my_set[4]
    
    
    # reshape data to fit in the CNN
    X_train = X_train.reshape((n_learn,512,31,1))
    X_validation = X_validation.reshape((3500-n_learn,512,31,1))
    X_test = X_test.reshape((3500,512,31,1))
    
    model = create_cnn((512,32,1),14)
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    model.fit(X_train,y_train,epochs = 20, batch_size = 32, validation_split = 0.2, verbose = 1)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_, verbose=1)
    print(test_loss)
    print(test_accuracy)

