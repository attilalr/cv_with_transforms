import time, sys

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 

from sklearn.model_selection import KFold

# local modules
from mlmodel import mlmodel
from mycross_val import mycross_val_score, my_nestedcross_val


### Classification case
X, y = make_classification(n_samples=80, n_features=8,
                           n_informative=4, n_redundant=0,
                           random_state=None, shuffle=True,
                           shift=None, scale=None,
                           )

random_seed = 1


# using the extended mlmodel class
# the methods are passed to the estimator obj but is acessible from clf.model
# check if the timming performance is the same as sklearn cross_val_score

print ('### Checking mlmodel class')
clf = mlmodel(RandomForestClassifier(random_state=random_seed),
            'Random Forest Classifier - A',
            )
print (clf)
print ('fit model with clf.fit')
clf.fit(X, y)

# create same cv for both
cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)


print ('### Checking computation time when running mycross_val_score against sklearn cross_val_score')
t = time.perf_counter()
scores_output1 = cross_val_score(clf.model, X, y, 
                      cv=cv, 
                      scoring='accuracy',
                      n_jobs=1)
print (f'cross_val_score: {scores_output1}')                      
print (f'Time of cross_val_score: {time.perf_counter()-t:.2f} s.')


# Restarting cv and RFC
cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
print ('### Checking mlmodel class')
clf = mlmodel(RandomForestClassifier(random_state=random_seed),
            'Random Forest Classifier - B',
            )
clf.fit(X, y)


print ('mycross_val_score...')
t = time.perf_counter()
scores_output2 = mycross_val_score(clf, X, y, 
                                  cv=cv, 
                                  scoring='accuracy', 
                                  )
print (f'mycross_val_score: {scores_output2}')
print (f'Time of mycross_val_score: {time.perf_counter()-t:.2f} s.')
print ('#\n')
#





#
print ('Lets check another scoring.')

print ('cross_val_score...')
t = time.perf_counter()
clf = mlmodel(RandomForestClassifier(random_state=random_seed),
            'Random Forest Classifier - A',
            )
clf.fit(X, y)
cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
scores_output1 = cross_val_score(clf.model, X, y, 
                      cv=cv, 
                      scoring='roc_auc', 
                      n_jobs=1)
print (f'cross_val_score: {scores_output1}')                      
print (f'Time of cross_val_score: {time.perf_counter()-t:.2f} s.')


print ('mycross_val_score...')
t = time.perf_counter()
clf = mlmodel(RandomForestClassifier(random_state=random_seed),
            'Random Forest Classifier - A',
            )
clf.fit(X, y)
cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
scores_output2 = mycross_val_score(clf, X, y, 
                                  cv=cv, 
                                  scoring='roc_auc', 
                                  )
print (f'mycross_val_score: {scores_output2}')
print (f'Time of mycross_val_score: {time.perf_counter()-t:.2f} s.')
print ('#\n')







# Testing transformation
print ('### Testing transformations')
print ('Passing standard scale...')
scaler = StandardScaler()
t = time.perf_counter()
scores_output = mycross_val_score(clf, X, y, 
                                scoring='accuracy',
                                cv=5, 
                                transform=scaler,
                                )
print (f'mycross_val_score: {scores_output}')
print (f'Time mycross_val_score: {time.perf_counter()-t:.2f} s.')
print ('#\n')

# testing transformation and train transformation
print('Passing standard scale and smote transformation...')
sm = SMOTE(random_state=42)
scaler = StandardScaler()
t = time.perf_counter()
scores_output = mycross_val_score(clf, X, y, 
                                cv=5, 
                                transform=scaler, 
                                train_transform=sm,
                                )
print (f'mycross_val_score: {scores_output}')
print (f'Time {time.perf_counter()-t:.2f} s.')
print ('#\n')




# Test nested cross validation
print ("### Test nested cross Validation")

# we need to put any number of mlmodels in it
# mlmodels are the usual models from sklearn, wrapped in a mlmodel class
est_list = list()
for i in range(10):
    est_list.append(
                    mlmodel(                                            # to create a mlmodel we need
                            RandomForestClassifier(max_depth=i+1),      # the sklearn model
                            f'Random Forest Classifier-maxdepth-{i+1}', # and a name
                            ), 
                    )


# execute the nested cv
list_best_models = my_nestedcross_val(est_list, X, y, 
                    cv=5,
                    score='accuracy',
                    cv_outer=3,
                    cv_inner=5,
                    n_jobs=2,
                    train_transform=None, train_transform_call=None,
                    transform=None, fit_transform_call=None, transform_call=None, 
                    )
print ('#\n')






### Regression case
print ('### Regression case')
X, y = make_regression(n_samples = 100, 
                       n_features = 5,
                       n_informative = 3, 
                       noise=1.0, 
                       shuffle=True, 
                       coef=False, 
                       random_state=None,
                       )



# using the extended mlmodel class
# the methods are passed to the estimator obj but is acessible from clf.model

print ('### Checking mlmodel class')
regr = mlmodel(RandomForestRegressor(random_state=random_seed),
            'Random Forest Regressor - A',
            )
print (regr)
print ('fit model with clf.fit')
regr.fit(X, y)




# check if the timming performance is the same as sklearn cross_val_score
print ('### Checking computation time when running mycross_val_score against sklearn cross_val_score')
t = time.perf_counter()
print ('cross_val_score:')
cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
scores_output = cross_val_score(regr.model, X, y, cv=5, n_jobs=1)
print (f'cross_val_score: {scores_output}')
print (f'Time of cross_val_score: {time.perf_counter()-t:.2f} s.')


# mycross_val_score
regr = mlmodel(RandomForestRegressor(random_state=random_seed),
            'Random Forest Regressor - B',
            )
print (regr)
print ('fit model with clf.fit')
regr.fit(X, y)
t = time.perf_counter()
cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
scores_output = mycross_val_score(regr, X, y, cv=5)
print (f'mycross_val_score: {scores_output}')
print (f'Time of mycross_val_score: {time.perf_counter()-t:.2f} s.')
print ('#\n')


