# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 21:17:53 2021

@author: -Tawsif
"""
from library import*
def models():
    
        clf=[]
        MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
               beta_2=0.999, early_stopping=False, epsilon=1e-08,
               hidden_layer_sizes=(13, 13), learning_rate='constant',
               learning_rate_init=0.001, max_iter=500, momentum=0.9,
               nesterovs_momentum=True, power_t=0.5, random_state=111,
               shuffle=False, solver='adam', tol=0.0001, validation_fraction=0.1,
               verbose=False, warm_start=False)
        clf.append(MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500))


        clf.append(LinearDiscriminantAnalysis())

        clf.append(xgb.XGBClassifier(
                        max_depth=85
                        ,learning_rate=0.9388440565186442,
                        min_split_loss= 0.0
                        ,reg_lambda=5.935581318908179
                        ,min_child_weight= 2.769401581888831
                        ,colsample_bylevel= 0.7878344729848824
                        ,colsample_bynode=0.4895496034538383
                        ,alpha= 7.9692927383000445
                        ,n_estimators=150
                        ,subsample = 0.2656532818978606
                        ,colsample_bytree = 0.8365485367400313))

        clf.append(RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                               max_depth=10, max_features='auto', max_leaf_nodes=None,
                               min_impurity_decrease=0.0,  
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=100,
                               n_jobs=None, oob_score=False, random_state=0,
                               verbose=0, warm_start=False))


        clf.append(LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                           intercept_scaling=1, l1_ratio=None, max_iter=100,
                           multi_class='multinomial', n_jobs=None, penalty='l2',
                           random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                           warm_start=False))


        clf.append(SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma='auto',
                kernel='linear', max_iter=100, probability=True, random_state=0,
                shrinking=True, tol=0.001, verbose=False))


        clf.append(ExtraTreesClassifier(n_estimators=100, max_depth=8, min_samples_split=10, random_state=0))

        clf.append(AdaBoostClassifier(n_estimators=100, random_state=0))

        clf.append(KNeighborsClassifier(n_neighbors=3))
        clf.append(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=10, random_state=0))

        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import VotingClassifier
        smote=SMOTE()
        #clf.append(VotingClassifier(estimators=[('MLP', model1), ('LLDR', model4),('LP', model7)], 
                               #voting='hard', weights=[2,4,5]))
        clff=['MLPClassifier','LinearDiscriminantAnalysis','XGBClassifier','RandomForestClassifier','LogisticRegression','SVM','ExtraTreesClassifier','AdaBoostClassifier','KNeighborsClassifier','GradientBoostingClassifier']
        #Result.to_csv
        return(clf,clff )