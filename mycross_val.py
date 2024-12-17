from os import lstat
from typing import List
from numpy.lib.arraysetops import isin
from numpy.lib.function_base import copy

import numpy as np
from mlmodel import mlmodel, mlclone

from sklearn.base import is_classifier, is_regressor, clone

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer, check_scoring
from sklearn.model_selection import check_cv

from scores import scores

from joblib import Parallel, logger
from joblib import parallel_backend
from sklearn.utils.parallel import delayed

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
    
    Scoring and predict_method must match.
    It means if you wanna use auc score you must provide the predict_method='pred_proba'.
    
    column_predict_proba:
        If you are using
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

    '''
    if callable(scoring):
        scorers = scoring
    elif scoring is None or isinstance(scoring, str):
        scorers = check_scoring(estimator, scoring)
    #else:
    #    scorers = _check_multimetric_scoring(estimator, scoring)
    '''

    ## scorers
    # scoring - names
    # scorer_list - functions
    if callable(scoring):
        scorer = scoring
        score_name = str(scoring)
    elif scoring == None: # default parameter
        scorer = check_scoring(estimator.model, None) # the default estimator score, as in cros_val_score 
        score_name = 'None'
    elif isinstance(scoring, str): # just one string for a metric
        scorer = get_scorer(scoring)
        score_name = scoring
    else:
        raise ValueError(f'scoring parameter unrecognized: {scoring}')


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
        
        scores_obj.register(score_name, scorer(estimator.model, X_test, y_true))

    return np.array(scores_obj[score_name])



def my_nestedcross_val(estimator_list: List, X, y, 
                    score='accuracy',
                    score_strategy_to_sort='nearest_to_zero_is_better', # higher_is_better, lower_is_better 
                    cv_outer=3,
                    cv_inner=5,
                    n_jobs=-1,
                    train_transform=None, train_transform_call=None,
                    transform=None, fit_transform_call=None, transform_call=None,
                    show_all_scores=False, 
                    hide_holdout_scores=False,
                    
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

    assert isinstance(X, np.ndarray), 'For now X must be a numpy array, if you are using pandas use df.values'
    assert isinstance(y, np.ndarray), 'For now y must be a numpy array, if you are using pandas use df.values'
                      
    assert len(estimator_list) > 0
    for estimator_ in estimator_list:
      assert isinstance(estimator_, mlmodel)

    kfold_outer = check_cv(cv=cv_outer, y=y, classifier=is_classifier(estimator_list[0]))

    lst_best_models = list()
    lst_best_scores_testing = list()
    lst_best_scores_holdout = list()

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
                    #scoring=score,
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
            lst_medias_scores.append(np.mean(result))
        
        if show_all_scores:
            for estimator_, mean_score_ in zip(estimator_list, lst_medias_scores):
                print (f'{estimator_.name} mean_score: {mean_score_:.3}')
            


        if score_strategy_to_sort == 'higher_is_better':
            proper_sort_fuction = np.argmax

        elif score_strategy_to_sort == 'lower_is_better':
            proper_sort_fuction = np.argmin

        elif score_strategy_to_sort == 'nearest_to_zero_is_better':

            def nearest_to_zero(x):
                return np.abs(np.array(x)-0).argmin()

            proper_sort_fuction = nearest_to_zero



        id_best_model = proper_sort_fuction(lst_medias_scores)
        name_best_model = estimator_list[id_best_model].name

        lst_best_models.append(estimator_list[id_best_model]) # guardar os melhores numa lista
        lst_best_scores_testing.append(lst_medias_scores[id_best_model])

        
        print (f'Best {score} score was {lst_medias_scores[id_best_model]:.3} of {name_best_model}, idx {id_best_model}')    

        clf = estimator_list[id_best_model].model    
        # 
        
        # 
        if transform:
            X_hold_train = transform.fit_transform(X_)
            X_hold_to_pred = transform.transform(X_holdout)

        else:

            X_hold_train = X_
            X_hold_to_pred = X_holdout


        clf.fit(X_hold_train, y_)
        y_true = y_holdout
        y_pred = clf.predict(X_hold_to_pred)
        
        #score_holdout = get_scorer(score)._score_func(y_true, y_pred)
        #score_holdout = get_scorer(score)(clf, y_true, y_pred)
        score_holdout = get_scorer(score)(clf, X_hold_to_pred, y_true)
        lst_best_scores_holdout.append(score_holdout)

        if not hide_holdout_scores:
            print(f'{score} of model {name_best_model} in holdout test set: {score_holdout:.3}')

    print ()
    print (f'Best {cv_outer} models:')

    if hide_holdout_scores:
        for estimator, testing_score, holdout_score in zip(lst_best_models, lst_best_scores_testing, lst_best_scores_holdout):
            print (f'{estimator.name}, testing score: {testing_score:.3}')
    else:
        for estimator, testing_score, holdout_score in zip(lst_best_models, lst_best_scores_testing, lst_best_scores_holdout):
            print (f'{estimator.name}, testing score: {testing_score:.3}, holdout score: {holdout_score:.3}')


    # Lets return a tuple of ((name1, estimator1), (name2, estimator2) ... ) 
    # for interoperability
    lst_best_models_ = list()

    for estimator in lst_best_models:
        lst_best_models_.append((estimator.name, estimator.model))

    return lst_best_models_
