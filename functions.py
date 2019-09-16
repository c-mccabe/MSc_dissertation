# Here we store some functions which will be used in the main script.

import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from skmultilearn.problem_transform import LabelPowerset

from scipy.io import mmread
from scipy.sparse import csr_matrix, vstack, hstack

from scipy import stats

from functools import partial


def LabelPowerset_Extra_predict(n_estimators, n_jobs, max_depth, min_samples_split, min_samples_leaf,
                                X_train, y_train, X_test, criterion = 'gini'):
    
    # this function fits an extremely randomised trees classifer with a 
    # label powerset problem transformation
    # and outputs its predictions on a test set.
    #
    # inputs: n_estimators      - the number of trees in the forest
    #         n_jobs            - the number of cores to use
    #         max_depth         - the maximum depth of trees in the forest
    #         min_samples_split - the minimum samples required to split a node
    #         min_samples_leaf  - the minimum samples required in a leaf node
    #         X_train, y_train  - the training data
    #         X_test            - test set to be predicted
    #         criterion         - the splitting criterion 
    #
    # outputs: predictions      - the predicted labels of the test set  
    
    classifier = LabelPowerset(
        classifier = ExtraTreesClassifier(n_estimators = n_estimators,
                                          min_samples_split = min_samples_split,
                                          n_jobs = n_jobs,
                                          min_samples_leaf = min_samples_leaf,
                                          max_depth = max_depth,
                                          criterion = criterion),
        require_dense = [False, True]
    )

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    return(predictions)


def LabelPowerset_RF_predict(n_estimators, n_jobs, max_depth, min_samples_split, min_samples_leaf,
                             X_train, y_train, X_test, criterion = 'gini'):
    
    # this function fits a random forest classifer with a lable powerset problem transformation
    # and outputs its predictions on a test set.
    #
    # inputs: n_estimators      - the number of trees in the forest
    #         n_jobs            - the number of cores to use
    #         max_depth         - the maximum depth of trees in the forest
    #         min_samples_split - the minimum samples required to split a node
    #         min_samples_leaf  - the minimum samples required in a leaf node
    #         X_train, y_train  - the training data
    #         X_test            - test set to be predicted
    #         criterion         - the splitting criterion 
    #
    # outputs: predictions      - the predicted labels of the test set
    
    classifier = LabelPowerset(
        classifier = RandomForestClassifier(n_estimators = n_estimators,
                                          min_samples_split = min_samples_split,
                                          n_jobs = n_jobs,
                                          min_samples_leaf = min_samples_leaf,
                                          max_depth = max_depth,
                                          criterion = criterion),
        require_dense = [False, True]
    )

    classifier.fit(X_train, y_train)
    
    predictions = classifier.predict(X_test)
    return(predictions)


def load_data():
    
    # this function loads the data
    
    X_train, X_test = mmread('data/X_train').tocsr(), mmread('data/X_test').tocsr()
    y_train, y_test = mmread('data/y_train').tocsr(), mmread('data/y_test').tocsr()
    
    return(X_train, X_test, y_train, y_test)

def objective_function(hyperparameters, X_train, X_test, y_train, y_test):
    
    # this function is the objective function which we will maximise in the 
    # Bayesian optimisation. it computes and stores the predictions of each base learner,
    # deletes the model from memory, and moves on to the next learner.
    # in this way we ensure that all models are not stored at once, which could lead
    # to memory issues for large data sets.
    #
    # inputs: hyperparameters  - the hyperparameters of the base models
    #         X_train, y_train - the training set
    #         X_test, y_test   - the test set
    #
    # output: accuracy         - the accuracy of the ensemble on the test set
    
    
    hyperparameters = np.array(hyperparameters)
    
    predictions_1 = LabelPowerset_RF_predict(n_estimators = hyperparameters[0][0].astype(int),
                                               max_depth = hyperparameters[0][1].astype(int),
                                               min_samples_split = hyperparameters[0][2].astype(int),
                                               min_samples_leaf = hyperparameters[0][3].astype(int),
                                               X_train = X_train, X_test = X_test,
                                               y_train = y_train, n_jobs = -1)
    
    predictions_2 = LabelPowerset_RF_predict(n_estimators = hyperparameters[0][4].astype(int),
                                               max_depth = hyperparameters[0][5].astype(int),
                                               min_samples_split = hyperparameters[0][6].astype(int),
                                               min_samples_leaf = hyperparameters[0][7].astype(int),
                                               X_train = X_train, X_test = X_test,
                                               y_train = y_train, n_jobs = -1)
    
    predictions_3 = LabelPowerset_RF_predict(n_estimators = hyperparameters[0][8].astype(int),
                                               max_depth = hyperparameters[0][9].astype(int),
                                               min_samples_split = hyperparameters[0][10].astype(int),
                                               min_samples_leaf = hyperparameters[0][11].astype(int),
                                               X_train = X_train, X_test = X_test,
                                               y_train = y_train, n_jobs = -1)
    
    predictions_4 = LabelPowerset_Extra_predict(n_estimators = hyperparameters[0][12].astype(int),
                                               max_depth = hyperparameters[0][13].astype(int),
                                               min_samples_split = hyperparameters[0][14].astype(int),
                                               min_samples_leaf = hyperparameters[0][15].astype(int),
                                               X_train = X_train, X_test = X_test,
                                               y_train = y_train, n_jobs = -1)
    
    predictions_5 = LabelPowerset_Extra_predict(n_estimators = hyperparameters[0][16].astype(int),
                                               max_depth = hyperparameters[0][17].astype(int),
                                               min_samples_split = hyperparameters[0][18].astype(int),
                                               min_samples_leaf = hyperparameters[0][19].astype(int),
                                               X_train = X_train, X_test = X_test,
                                               y_train = y_train, n_jobs = -1)
    
    predictions_6 = LabelPowerset_Extra_predict(n_estimators = hyperparameters[0][20].astype(int),
                                               max_depth = hyperparameters[0][21].astype(int),
                                               min_samples_split = hyperparameters[0][22].astype(int),
                                               min_samples_leaf = hyperparameters[0][23].astype(int),
                                               X_train = X_train, X_test = X_test,
                                               y_train = y_train, n_jobs = -1)
    
    predictions_7 = LabelPowerset_Extra_predict(n_estimators = hyperparameters[0][24].astype(int),
                                               max_depth = hyperparameters[0][25].astype(int),
                                               min_samples_split = hyperparameters[0][26].astype(int),
                                               min_samples_leaf = hyperparameters[0][27].astype(int),
                                               X_train = X_train, X_test = X_test,
                                               y_train = y_train, n_jobs = -1)
    
    # here we combine the predictions of the base models into an array, over which we will perform
    # a majority vote.
    
    full_predictions = np.array([predictions_1.todense(), predictions_2.todense(), predictions_3.todense(),
                                predictions_4.todense(), predictions_5.todense(), predictions_6.todense(),
                                predictions_7.todense() ])
    
    majority_vote = stats.mode(full_predictions)[0][0]
    
    # now calculate the accuracy and print results
    
    accuracy = accuracy_score(majority_vote, y_test)
    
    print('hyperparameters: {}'.format(hyperparameters[0]))
    print('accuracy: {}'.format(accuracy))
    
    return(accuracy)









