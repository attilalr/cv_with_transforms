import time

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 

# local modules
from mlmodel import mlmodel
from mycross_val import mycross_val_score, mycross_val_predict, my_nestedcross_val_predict


X, y = make_classification(n_samples=200, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=True)

# using the extended mlmodel class
# the methods are passed to the estimator obj but is acessible from clf.model
clf = mlmodel(RandomForestClassifier(),
            'Random Forest Classifier - A',
            )
print (clf)
clf.fit(X, y)

# check if it is working
print (clf.feature_importances_)
print (clf)

# check if the timming performance is the same as sklearn cross_val_score
t = time.perf_counter()
print ('cross_val_score:')
print(cross_val_score(clf.model, X, y, cv=5, n_jobs=1))
print (f'Time {time.perf_counter()-t:.2f} s.')

t = time.perf_counter()
scores_output = mycross_val_score(clf, X, y, cv=5)
print (f'mycross_val_score: {scores_output}')
print (f'Time {time.perf_counter()-t:.2f} s.')


# testing transformation
print ('### Testing transformations')
print ('Passing standard scale...')
scaler = StandardScaler()
t = time.perf_counter()
scores_output = mycross_val_score(clf, X, y, 
                                cv=5, 
                                transform=scaler,
                                )
print (f'mycross_val_score: {scores_output}')
print (f'Time {time.perf_counter()-t:.2f} s.')


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

# testing transformation and train transformation
print('### Testing cross val predict')
y_pred = mycross_val_predict(clf, X, y, 
                    cv=5,
                    method='predict',
                    train_transform=None, train_transform_call=None,
                    transform=None, fit_transform_call=None, transform_call=None, 
                    )
print (y_pred)


print('### Testing cross val predict_proba')
y_pred_proba = mycross_val_predict(clf, X, y, 
                    cv=5,
                    method='predict_proba',
                    train_transform=None, train_transform_call=None,
                    transform=None, fit_transform_call=None, transform_call=None, 
                    )
print (y_pred_proba)

# Test nested cross alidation
print ("# Test nested cross Validation")
est_list = list()
for i in range(10):
    est_list.append(mlmodel(RandomForestClassifier(),
            f'Random Forest Classifier-{i}',
            ))

list_best_models = my_nestedcross_val_predict(est_list, X, y, 
                    cv=5,
                    method='predict',
                    score='accuracy',
                    cv_outer=3,
                    cv_inner=5,
                    n_jobs=2,
                    train_transform=None, train_transform_call=None,
                    transform=None, fit_transform_call=None, transform_call=None, 
                    )