import numpy as np 
import matplotlib.pyplot as plt 
import os

def deal_outliers(dataset,Z_th):
    """
    Replace Outliers by missing values that are then going to be replaced by mean of next and previous
    points.

    args : 
    dataset : a sensor txt file. 
    Z_th : the z score threshold to consider if a point is an outlier or not.
    """
    nb_changed = 0
    
    for line in range(len(dataset)):
        local_change = 0
        curr_mean = np.mean(dataset[line])
        if curr_mean > -10000:
            curr_std = np.std(dataset[line])
            if curr_std != 0:
                for j in range(len(dataset[line])): 
                
                    Z_score = (dataset[line][j] - curr_mean)/(curr_std)
                    if abs(Z_score) > Z_th:
                        nb_changed += 1
                        #Using average of neighbors point : 
                        """local_change+=1
                        if j > 0 and j < len(dataset[line]) - 1:
                            dataset[line][j] = (dataset[line][j-1]+dataset[line][j+1])/2
                        if j == 0:
                            dataset[line][j] = (dataset[line][j+1] + dataset[line][j+2])/2
                        if j == len(dataset[line]) - 1:
                            dataset[line][j] = (dataset[line][j-1] + dataset[line][j-2])/2"""
                        #Using median (best I think). 
                        dataset[line][j] = np.median(dataset[line])
    print(f"number of values changed in this dataset = {nb_changed} \n")

if __name__ == '__main__':
    print("main")
    LS_path = os.path.join('./', 'LS')
    dataset = np.loadtxt(os.path.join(LS_path, 'LS_sensor_{}.txt'.format(5)))
    plt.plot(dataset[815])
    plt.show()
    deal_outliers(dataset,1.5)
    plt.plot(dataset[815])
    plt.show()