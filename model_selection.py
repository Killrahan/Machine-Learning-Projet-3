import numpy as np

from csv_diff import compare_csv_files
from toy_script import write_submission
from fourier import build_dataset,CV_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from subjectSplit import SubjectSplit


if __name__ == "__main__":
    
    # build dataset with Simple Imputer, stats and fourier
    dataset = build_dataset(5)
    
    subject_splitter = SubjectSplit(dataset[0])
    
    X_train = np.concatenate(dataset[0])
    y_train = np.concatenate(dataset[1])
    X_test = dataset[2]
    
    # set files for comparison
    
    file1 = 'test_labels_sorted.csv'
    file2 = 'example_submission.csv'
    
    # kNN classfier ==========================
    
    print("kNN")
    
    params_grid = {
        'n_neighbors' : np.arange(1,201).tolist()
    }
    
    grid = GridSearchCV(estimator=KNeighborsClassifier(),param_grid=params_grid,cv=subject_splitter,n_jobs=-1)
    grid.fit(X_train,y_train)
    
    best_param = grid.best_params_
    best_score = grid.best_score_
    
    print("best score :",best_score)
    print("best param :", best_param)
    
    clf = KNeighborsClassifier(n_neighbors=best_param['n_neighbors'])
    clf.fit(X_train,y_train)
    
    y_pred = clf.predict(X_test)
    write_submission(y_pred)
    
    # print errors in the terminal
    compare_csv_files(file1, file2)
    
    
    # DT classfier ==========================
    
    print("DT")
    
    params_grid = {
        'max_depth' : np.arange(1,32).tolist().append(None)
    }
    
    grid = GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=params_grid,cv=subject_splitter,n_jobs=-1)
    grid.fit(X_train,y_train)
    
    best_param = grid.best_params_
    best_score = grid.best_score_
    
    print("best score :",best_score)
    print("best param :", best_param)
    clf = DecisionTreeClassifier(max_depth=best_param['max_depth'])
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)
    write_submission(y_pred)

    # print errors in the terminal
    compare_csv_files(file1, file2)
    
    # Multi-layer perceptron ================
    clf = MLPClassifier()
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)
    write_submission(y_pred)

    # print errors in the terminal
    compare_csv_files(file1, file2)
    
    # Random forest classifier ==============
    
    print("RFC")
    
    params_grid = {
        'max_depth' : np.arange(1,32).tolist().append(None),
        'max_features' : ['sqrt','log2',None]
    }
    
    grid = GridSearchCV(estimator=KNeighborsClassifier(),param_grid=params_grid,cv=subject_splitter,n_jobs=-1)
    grid.fit(X_train,y_train)
    
    best_param = grid.best_params_
    best_score = grid.best_score_
    
    print("best score :",best_score)
    print("best param :", best_param)
    clf = RandomForestClassifier(n_estimators= 1500, n_jobs=-1,max_depth=best_param['max_depth'],max_features=best_param['max_features'])
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)
    write_submission(y_pred)

    # print errors in the terminal
    compare_csv_files(file1, file2)
    