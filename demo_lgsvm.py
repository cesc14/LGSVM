import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn import svm
import time

from lgsvm import LGSVM

"""

Demo example concerning the usage of the LGSVM classifier.

Authors: Marchetti F. & Perracchione E.

"""

# =============================================================================
# DATASET GENERATION
# =============================================================================

X, y = make_classification(n_samples = 30000, n_features=2, n_redundant=0,\
                           n_informative=1, n_clusters_per_class=1,\
                               random_state = 42)

the_scaler = preprocessing.StandardScaler()
X = the_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,\
                                                stratify = y, random_state=42)

# =============================================================================
# LGSVM CLASSIFICATION
# =============================================================================

model = LGSVM(metric = 'cosine', approach = 'grid', n_centers = 21,\
              svm_model = svm.SVC(kernel='rbf'))

t = time.time()

model.fit(X_train, y_train.ravel())

y_predict = model.predict(X_test)

print("Elapsed time:", np.around(time.time() - t,2),'s')
print('F1-scores:',np.around(f1_score(y_test,y_predict,average='weighted'),3))
print('\n')


# =============================================================================
# PLOT
# =============================================================================
    
    
fig = plt.figure()

pos_a = np.where(y_train==0)[0]
pos_b = np.where(y_train==1)[0]

X_a = X_train[pos_a]
X_b = X_train[pos_b]

plt.scatter(X_a[:, 0], X_a[:, 1],facecolors="None", marker='o',\
            edgecolors='black', s=30)

plt.scatter(X_b[:, 0], X_b[:, 1], marker='s', color="black", s=30)

pos_a_ = np.where(y_predict==0)[0]
pos_b_ = np.where(y_predict==1)[0]

X_a_ = X_test[pos_a_]
X_b_ = X_test[pos_b_]

plt.scatter(X_a_[:, 0], X_a_[:, 1],facecolors="blue", marker='o',\
            edgecolors='black', s=30)

plt.scatter(X_b_[:, 0], X_b_[:, 1], marker='s', color="red", s=30)