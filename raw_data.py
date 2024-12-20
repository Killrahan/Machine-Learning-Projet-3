import numpy as np
import os

from Resolve import get_subject_sensors

from sklearn.ensemble import RandomForestClassifier
from toy_script import write_submission


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
    subject_indexes = []


    # subject_indexes[i] contains the indexes of subject_id = i
    for i in range(1, nb_tot+1):
        curr_indexes = get_subject_sensors(subject_array, i)
        subject_indexes.append(curr_indexes)

    # Build sets :
    
    X = []
    for i in range(nb_tot):
        X.append(np.zeros((len(subject_indexes[i]), (len(f_indexes)*512))))

    X_test = np.zeros((3500, (len(f_indexes) * 512)))

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


    for i in range(nb_tot):
        index = 0
        for f in f_indexes:
            k = 0
            print(f"f = {f} \n")
            for line_index in subject_indexes[i]:
                X[i][:, (index)*512:(index+1) *
                        512][k] = data_array[index][int(line_index)] #set_stat(data_array[index][int(line_index)])
                k += 1


            for line in range(3500):
                X_test[:, (index)*512:(index+1) *
                        512][line] = data_test_array[index][line]#set_stat(data_test_array[index][line])

            index += 1

    return X, y, X_test

def build_subsets(X,y,validation_index,split = True):
    
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
        

    return X_train,y_train,X_validation,y_validation

def score(X,y,i,n_estimators,max_depth,max_features):
    
    X_train, y_train, X_validation, y_validation = build_subsets(X,y,i)
    clf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features)
    clf.fit(X_train,y_train)
    
    return clf.score(X_validation,y_validation)

def CV_score(dataset,n_estimators,max_depth,max_features):
    
    
    X = dataset[0]
    y = dataset[1]
    X_test = dataset[2]
    
    K = len(X)
    
    scores = np.zeros(K)
    
    for i in range(K):
        print(i)
        score_i = score(X,y,i,n_estimators,max_depth,max_features)
        scores[i] = score_i
        print(score_i)
    
    return np.average(scores)
        
if __name__ == "__main__":
    dataset = build_dataset(5)
    print(CV_score(dataset,100,None,'sqrt'))
    
    X_train = np.concatenate(dataset[0])
    y_train = np.concatenate(dataset[1])
    
    X_test = dataset[2]
    
    clf = RandomForestClassifier(n_estimators = 1500, n_jobs=-1)
    clf.fit(X_train,y_train)
    
    y_predict = clf.predict(X_test)
    write_submission(y_predict)