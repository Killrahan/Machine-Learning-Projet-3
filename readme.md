# What was done in this branch?

## Main idea

The main idea here is to reduce the dimentionality of the estimators by using a similarity metric between times series for each sensor. 

Rather than using all the points in all the time series as inputs, our idea is to use the whole time serie of a sensor as an input. 

## Process

1. First, we preprocess the data
2. We arrange the data to use whole time series as a feature
3. We train an estimator on that data

## Tests done and results

### 1. kNN

kNN yielded the same results as if we did not change the algorithm

### 2. tbd
