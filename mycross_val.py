from os import lstat
from typing import List
from numpy.lib.arraysetops import isin
from numpy.lib.function_base import copy

import numpy as np
from mlmodel import mlmodel, mlclone

from sklearn.base import is_classifier, is_regressor, clone

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from sklearn.model_selection import check_cv

from scores import scores

from joblib import Parallel, logger
from joblib import parallel_backend
from sklearn.utils.fixes import delayed

def get_transformations_calls(
                            train_transform=None, # the object training set exclusive (ex. SMOTE)
                            train_transform_call=None, # customization of the call
                            transform=None, # the fit_transform object (ex. standardization)
                            fit_transform_call=None, # customization of the fit_transform call (training set)
                            transform_call=None, # customization of the transform call (applied to test set)
                            ):

    # train_transform call customization
    # usual for imblearn is .fit_resample
    if train_transform:
        if isinstance(train_transform_call, str):
            method_fit_resample = getattr(train_transform, train_transform_call)
        elif train_transform:
            method_fit_resample = getattr(train_transform, 'fit_resample')
    else:
        method_fit_resample = None

    if transform:
        # transformation for training/test set, customizing call
        if isinstance(fit_transform_call, str):
            method_fit_transform = getattr(transform, fit_transform_call)
        elif transform:
            method_fit_transform = getattr(transform, 'fit_transform')

        if isinstance(transform_call, str):
            method_transform = getattr(transform, transform_call)
        elif transform:
            method_transform = getattr(transform, 'transform')
    else:
        method_fit_transform = None
        method_transform = None

    # returns
    return method_fit_resample, method_fit_transform, method_transform


def mycross_val_score(estimator, X, y, 
                    scoring=None,
                    cv=5,
                    train_transform=None, train_transform_call=None,
                    transform=None, fit_transform_call=None, transform_call=None,
                    ) -> np.array:
    '''
    Perform a cross-validation and return a score vector.

    The transformations objectives are divided in two: the training set (for each fold) 
        exclusive transformations and the fit_transform in the training and transform 
        on the test set.

    First one:
    train_transform: transformation method exclusive to the training set for each fold.
        Intended to perform oversampling or synthetic-like data generation as SMOTE
        techniques. The train_transform is applied first. Uses fit_resample method.
    train_transform_call: customization of the call, the default is 'fit_resample'.

    Second:
    transform: define the transformation method which will be fit and transform
        to the training data for each fold. The test set is transformed using 
        the training fit state. Example: standardization. The default are 'fit_transform' 
        and 'transform' methods. 
    fit_transform_call: Customization in the fit_transform name call, the default is 
        'fit_transform'.
    transform_call: customization of the transform call, default is 'transform'.
    '''

    method_fit_resample, method_fit_transform, method_transform = get_transformations_calls(
                            train_transform=train_transform, # the object training set exclusive (ex. SMOTE)
                            train_transform_call=train_transform_call, # customization of the call
                            transform=transform, # the fit_transform object (ex. standardization)
                            fit_transform_call=fit_transform_call, # customization of the fit_transform call (training set)
                            transform_call=transform_call, # customization of the transform call (applied to test set)
                            )


    # original parameters of cross_val_score
    #groups=None, 
    #scoring=None, 
    #cv=None, 
    #n_jobs=None, 
    #verbose=0, 
    #fit_params=None, 
    #pre_dispatch='2*n_jobs', 
    #error_score=nan,
    
    kfold = check_cv(cv=cv, y=y, classifier=is_classifier(estimator))

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

def mycross_val_predict(estimator, X, y, 
                    cv=5,
                    method='predict',
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

    assert y.ndim == 1, 'The function support only single output.'

    method_fit_resample, method_fit_transform, method_transform = get_transformations_calls(
                            train_transform=train_transform, # the object training set exclusive (ex. SMOTE)
                            train_transform_call=train_transform_call, # customization of the call
                            transform=transform, # the fit_transform object (ex. standardization)
                            fit_transform_call=fit_transform_call, # customization of the fit_transform call (training set)
                            transform_call=transform_call, # customization of the transform call (applied to test set)
                            )

    # original parameters of cross_val_score
    #groups=None, 
    #scoring=None, 
    #cv=None, 
    #n_jobs=None, 
    #verbose=0, 
    #fit_params=None, 
    #pre_dispatch='2*n_jobs', 
    #error_score=nan,
    
    kfold = check_cv(cv=cv, y=y, classifier=is_classifier(estimator))

    # set method to predict and initialize the prediction vector
    if method == 'predict_proba':
        y_pred_all = np.empty((y.size, np.unique(y).size))
    elif method == 'predict':
        y_pred_all = np.empty_like(y)

    assert method == 'predict_proba' or method == 'predict', 'Supported methods'
    method_predict = getattr(estimator, method)



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
        if method == 'predict_proba':
            y_pred_all[test_index, :] = method_predict(X_test)
        elif method == 'predict':
            y_pred_all[test_index] = method_predict(X_test)

    return y_pred_all


def my_nestedcross_val(estimator_list: List, X, y, 
                    cv=5,
                    method='predict',
                    score='accuracy',
                    cv_outer=3,
                    cv_inner=5,
                    n_jobs=-1,
                    train_transform=None, train_transform_call=None,
                    transform=None, fit_transform_call=None, transform_call=None, 
                    ) -> List:
    '''
    Perform a cross-validation and return a cv-outer best models list.

    train_transform: transformation exclusive to the training set for each fold.
        Intended to perform oversampling or synthetic-like data generation as SMOTE
        techniques. The train_transform is applied first. Uses fit_resample method.
    transform: define a transformation which will be fit and transform
        to the training data for each fold. The test set is transformed using 
        the training fit state. Example: standardization. Uses fit_transform and transform
        methods.
    '''
    assert len(estimator_list) > 0
    assert isinstance(estimator_list[0], mlmodel)

    kfold_outer = check_cv(cv=cv_outer, y=y, classifier=is_classifier(estimator_list[0]))

    lst_best_models = list()
    for j, (train_index_outer, test_index_outer) in enumerate(kfold_outer.split(X, y)):

        print (f'Outer Fold {j+1} of a total {cv_outer}...')
        
        X_ = X[train_index_outer]
        y_ = y[train_index_outer] 

        X_holdout = X[test_index_outer]
        y_holdout = y[test_index_outer]
        
        kfold_inner = check_cv(cv=cv_inner, y=y, classifier=is_classifier(estimator_list[0]))

        '''
        # old serial for loop 
        for estimator in estimator_list:
            estimator.scores.register(score, np.mean(mycross_val_score(estimator, X_, y_, 
                                                    scoring=score,
                                                    cv=cv_inner,
                                                    train_transform=train_transform, 
                                                    train_transform_call=train_transform_call,
                                                    transform=transform, fit_transform_call=fit_transform_call, 
                                                    transform_call=transform_call,
                                                    )[score]),
                                    )
            print (estimator)
        '''

        verbose = 2
        pre_dispatch = '2*n_jobs'

        with parallel_backend('loky'):
            parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)

            results = parallel(
                delayed(mycross_val_score)(
                    mlclone(estimator),
                    X_,
                    y_,
                    scoring=score,
                    cv=cv_inner,
                    train_transform=train_transform, 
                    train_transform_call=train_transform_call,
                    transform=transform, fit_transform_call=fit_transform_call, 
                    transform_call=transform_call,
                )
                for estimator in estimator_list
            )

        #for estimator, result in zip(estimator_list, results):
        #    estimator.scores.register(score, np.mean(result[score]))



        lst_medias_scores = list()
        for estimator, result in zip(estimator_list, results):
            lst_medias_scores.append(np.mean(result[score]))
        
            
            
        name_best_model = estimator_list[np.argmax(lst_medias_scores)].name
        id_best_model = np.argmax(lst_medias_scores)
        lst_best_models.append(estimator_list[np.argmax(lst_medias_scores)]) # guardar os melhores numa lista

        print (f'Best {score} score was {np.max(lst_medias_scores):.3f} of {name_best_model}, idx {id_best_model}')    

        clf = estimator_list[np.argmax(lst_medias_scores)].model    
        # 
        
        # 
        clf.fit(X_, y_)
        y_true = y_holdout
        y_pred = clf.predict(X_holdout)

        print(f'{score} of model {name_best_model} in holdout test set: {get_scorer(score)._score_func(y_true, y_pred):.3f}')

    print ()
    print (f'Best {cv_outer} models:')
    for estimator in lst_best_models:
        print (f'{estimator.name}')

    return lst_best_models
