from subjectSplit import SubjectSplit
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

dataset = [
    [[[1,1],[1,1],[1,1]],[[2,2]],[[3,3],[3,3],[3,3]]],
    [[1,1,1],[2],[3,3,3]],
    []
    ]
    
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
    'n_neighbors' : np.arange(1,4).tolist()
}

grid = GridSearchCV(estimator=KNeighborsClassifier(),param_grid=params_grid,cv=subject_splitter,n_jobs=-1)
grid.fit(X_train,y_train)

best_param = grid.best_params_
best_score = grid.best_score_

print("best score :",best_score)
print("best param :", best_param)

clf = KNeighborsClassifier()
clf.fit(X_train,y_train)

