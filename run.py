from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 

from mlmodel import mlmodel
from mycross_val import mycross_val_score

from sklearn.model_selection import cross_val_score

import time

X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)

clf = mlmodel(RandomForestClassifier(),
            'Random Forest Classifier - 1',
            )

clf.fit(X, y)

print (clf.feature_importances_)
print (clf)

t = time.time()
print ('cross_val_score:')
print(cross_val_score(clf.model, X, y, cv=5, n_jobs=1))
print (f'Tempo {time.time()-t:.2f} s.')

t = time.time()
scores_output = mycross_val_score(clf, X, y, cv=5)
print (f'mycross_val_score: {scores_output}')
print (f'Tempo {time.time()-t:.2f} s.')


print ('Passing standard scale...')

scaler = StandardScaler()
t = time.time()
scores_output = mycross_val_score(clf, X, y, cv=5, transform=scaler)
print (f'mycross_val_score: {scores_output}')
print (f'Tempo {time.time()-t:.2f} s.')


print('Passing standard scale and smote transformation...')

sm = SMOTE(random_state=42)
#X_res, y_res = sm.fit_resample(X, y)

scaler = StandardScaler()
t = time.time()
scores_output = mycross_val_score(clf, X, y, cv=5, transform=scaler, train_transform=sm)
print (f'mycross_val_score: {scores_output}')
print (f'Tempo {time.time()-t:.2f} s.')