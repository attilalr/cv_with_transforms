from numpy.lib.arraysetops import isin
from numpy.lib.function_base import copy
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from scores import scores

import numpy as np

def mycross_val_score(estimator, X, y, 
                    scoring=None,
                    cv = 5,
                    train_transform=None, train_transform_call=None,
                    transform=None, fit_transform_call=None, transform_call=None, 
                    ) -> np.array:
    '''
    Perform a cross-validation and return a score vector.

    train_transform: transformation exclusive to the training set for each fold.
        Intended to perform oversampling or synthetic-like data generation as SMOTE
        techniques. The train_transform is applied first. Uses fit_resample method.
    transform: define a transformation which will be fit and transform
        to the training data for each fold. The test set is transformed using 
        the training fit state. Example: standardization. Uses fit_transform and transform
        methods.
    '''

    # train_transform call customization
    # usual for imblearn is .fit_resample
    if isinstance(train_transform_call, str):
        method_fit_resample = getattr(train_transform, train_transform_call)
    elif train_transform:
        method_fit_resample = getattr(train_transform, 'fit_resample')

    # transformation for training/test set, customizing call
    if isinstance(fit_transform_call, str):
        method_fit_transform = getattr(transform, fit_transform_call)
    elif transform:
        method_fit_transform = getattr(transform, 'fit_transform')

    if isinstance(transform_call, str):
        method_transform = getattr(transform, transform_call)
    elif transform:
        method_transform = getattr(transform, 'transform')

    #groups=None, 
    #scoring=None, 
    #cv=None, 
    #n_jobs=None, 
    #verbose=0, 
    #fit_params=None, 
    #pre_dispatch='2*n_jobs', 
    #error_score=nan,
    
    if cv == None:
        cv = 5

    if is_classifier(estimator) and isinstance(cv, int):
        kfold = StratifiedKFold(n_splits=cv, shuffle=False)
    elif not is_classifier(estimator) and isinstance(cv, int):
        kfold = KFold(n_splits=cv, shuffle=False)
    else:
        # iterators and others not implemented
        assert isinstance(cv, int)

    ## scorers
    # scoring - names
    # scorer_list - functions
    scorer_list = list()
    if scoring == None: # default parameter
        scoring = 'accuracy'
        scorer_list.append(get_scorer(scoring)._score_func)
        scoring = [scoring] # put in a list to iterate after
    elif isinstance(scoring, str): # just one string for a metric
        scorer_list.append(get_scorer(scoring)._score_func)
        scoring = [scoring] # put in a list to iterate after
    elif isinstance(scoring, list): # if we have more than one score
        for score_name in scoring:
            scorer_list.append(get_scorer(score_name)._score_func)

    scores_obj = scores()
    #

    for train_index, test_index in kfold.split(X, y):

        X_train = X[train_index].copy()
        y_train = y[train_index] 

        X_test = X[test_index].copy()
        y_test = y[test_index]

        if transform:
            # fit/apply the transformation in training set
            # apply to the test set
            #X_train = transform.fit_transform(X_train)
            #X_test = transform.transform(X_test)
            X_train = method_fit_transform(X_train)
            X_test = method_transform(X_test)
        
        if train_transform:
            # smote-like techniques only applied to the training set.
            #X_train = train_transform.fit_resample(X_train, y_train)
            X_train, y_train = method_fit_resample(X_train, y_train)

        estimator.fit(X_train, y_train)
        y_true = y_test
        y_pred = estimator.predict(X_test)

        for score_name, scorer in zip(scoring, scorer_list):
            scores_obj.register(score_name, scorer(y_true, y_pred))

    return scores_obj
