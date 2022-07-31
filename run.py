import time, sys

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 

from sklearn.model_selection import KFold

# local modules
from mlmodel import mlmodel
from mycross_val import mycross_val_score, mycross_val_predict, my_nestedcross_val


### Classification case
X, y = make_classification(n_samples=80, n_features=8,
                           n_informative=4, n_redundant=0,
                           random_state=None, shuffle=True,
                           shift=None, scale=None,
                           )




# using the extended mlmodel class
# the methods are passed to the estimator obj but is acessible from clf.model

print ('### Checking mlmodel class')
clf = mlmodel(RandomForestClassifier(),
            'Random Forest Classifier - A',
            )
print (clf)
print ('fit model with clf.fit')
clf.fit(X, y)

# check if it is working
print ('clf.feature_importances')
print (clf.feature_importances_)
print (clf)
print ('#\n')






# check if the timming performance is the same as sklearn cross_val_score

# create same cv for both
cv = KFold(n_splits=5, shuffle=False, random_state=None)

scoring = 'accuracy'

print ('### Checking computation time when running mycross_val_score against sklearn cross_val_score')
t = time.perf_counter()
scores_output1 = cross_val_score(clf.model, X, y, 
                      cv=cv, 
                      scoring=scoring,
                      n_jobs=1)
print (f'cross_val_score: {scores_output1}')                      
print (f'Time of cross_val_score: {time.perf_counter()-t:.2f} s.')

cv = KFold(n_splits=5, shuffle=False, random_state=None)

print ('mycross_val_score...')
t = time.perf_counter()
scores_output2 = mycross_val_score(clf, X, y, 
                                  cv=cv, 
                                  scoring=scoring, 
                                  #predict_method='predict',
                                  )
print (f'mycross_val_score: {scores_output2}')
print (f'Time of mycross_val_score: {time.perf_counter()-t:.2f} s.')
print ('#\n')
#

sys.exit(0)








# Testing transformation
print ('### Testing transformations')
print ('Passing standard scale...')
scaler = StandardScaler()
t = time.perf_counter()
scores_output = mycross_val_score(clf, X, y, 
                                scoring='accuracy',
                                predict_method='predict',
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

'''
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
'''                    
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
regr = mlmodel(RandomForestRegressor(),
            'Random Forest Regressor - A',
            )
print (regr)
print ('fit model with clf.fit')
regr.fit(X, y)

# check if it is working
print ('Features importantes:')
print (regr.feature_importances_)
print (regr)
print ('#\n')





# check if the timming performance is the same as sklearn cross_val_score
print ('### Checking computation time when running mycross_val_score against sklearn cross_val_score')
t = time.perf_counter()
print ('cross_val_score:')
print(cross_val_score(regr.model, X, y, cv=5, n_jobs=1))
print (f'Time of cross_val_score: {time.perf_counter()-t:.2f} s.')

t = time.perf_counter()
scores_output = mycross_val_score(regr, X, y, cv=5)
print (f'mycross_val_score: {scores_output}')
print (f'Time of mycross_val_score: {time.perf_counter()-t:.2f} s.')
print ('#\n')


