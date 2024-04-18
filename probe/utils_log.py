from common.utils import *

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

def cross_validate_model(train_folds,
                         test_folds,
                         class_weights_folds, 
                         layer, 
                         n_folds=5):
    scores = []
    for i in range(n_folds):
        #import pdb; pdb.set_trace()
        X_train, y_train = train_folds[i]
        X_train = X_train[:,layer,:]
        X_test, y_test = test_folds[i]
        X_test = X_test[:,layer,:]        
        model = LogisticRegression(max_iter=5000, class_weight=class_weights_folds[i])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        scores.append(accuracy_score(y_test, predictions))
    
    return np.mean(scores), scores
