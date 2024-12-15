import numpy as np
import os

import toy_script

from sklearn.model_selection import GridSearchCV

from subjectSplit import SubjectSplit
from Resolve import get_subject_sensors
from Resolve import set_stat

from knn_imputation import fill as fill_knn

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier

def build_dataset(nb_tot):
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


    LS_path = os.path.join('./', 'LS')
    TS_path = os.path.join('./', 'TS')

    
    # equivalent to the for loop in Resolve.py with all sensors kept
    f_indexes = np.arange(2,33)
    

    # Getting indexes associated to subjects ids
    subject_array = np.loadtxt(os.path.join(LS_path, 'subject_Id.txt'))
    
    # create indexes as a list of empty np arrays
    indexes = []


    # indexes[i] contains the indexes of subject_id = i
    for i in range(1, nb_tot+1):
        curr_indexes = get_subject_sensors(subject_array, i)
        indexes.append(curr_indexes)

    # Build sets and fill missing data :
    
    X = []
    for i in range(nb_tot):
        X.append(np.zeros((len(indexes[i]), (len(f_indexes)*521))))

    X_test = np.zeros((3500, (len(f_indexes) * 521)))

    y_data = np.loadtxt(os.path.join(LS_path, 'activity_Id.txt'))
    y = []
    
    # y[i] contains all the elements of activity_id where subject_id = i
    for i in range(nb_tot):
        y_i = []
        for j in indexes[i]:
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

    data_array = fill_knn(data_array)

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
            for line_index in indexes[i]:
                X[i][:, (index)*521:(index+1) *
                        521][k] = set_stat(data_array[index][int(line_index)])
                k += 1


            for line in range(3500):
                X_test[:, (index)*521:(index+1) *
                        521][line] = set_stat(data_test_array[index][line])

            index += 1

    return X, y, X_test

if __name__ == "__main__":
    dataset = build_dataset(5)
    
    X = dataset[0]
    y = dataset[1]
    X_test = dataset[2]
    
    X_flat = np.concatenate(X)
    y_flat = np.concatenate(y)
    
    param_grid = {'max_depth' : np.arange(1,33), 'max_features' : ['sqrt','log2',None]}
    clf = RandomForestClassifier()
    
    subject_splitter = SubjectSplit(5)
    
    grid_search = GridSearchCV(clf,param_grid,cv = subject_splitter,n_jobs=None)
    grid_search.fit(X_flat,y_flat)
    
    print("Best parameters : ",grid_search.best_params_)
    print("Best score : ",grid_search.best_score_)
    
    best_params = grid_search.best_params_
    max_depth = best_params['max_depth']
    max_feat = best_params['max_features']
    
    clf = RandomForestClassifier(n_estimators=1500,max_depth=max_depth,max_features=max_feat)
    clf.fit(X_flat,y_flat)
    y_pred = clf.predict(X_test)
    toy_script.write_submission(y_pred)
    
