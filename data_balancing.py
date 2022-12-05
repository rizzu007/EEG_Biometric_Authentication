import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

def data_balancing(x_tr, x_ts,y_tr,y_ts):
    smote=SMOTE()
    xtrain=[]
    xtest=[]
    ytrain=[]
    ytest=[]
    X_train, X_test =x_tr, x_ts
    y_train, y_test = y_tr,y_ts
    x1,y1= smote.fit_resample(X_train, y_train)
    xtrain.append(x1)
    xtest.append(X_test)
    ytrain.append(y1)
    ytest.append(y_test)
    d={'data':(xtrain,xtest,ytrain,ytest)}
    return d