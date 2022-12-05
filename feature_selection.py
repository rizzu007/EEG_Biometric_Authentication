
from library import*
def feature_selection(x,y):
        X_train=x
        y_train=y
        mdl=[]
        mdl.append(xgb.XGBClassifier(
                        max_depth=4
                        ,learning_rate=0.2
                        ,reg_lambda=1
                        ,n_estimators=50
                        ,subsample = 0.9
                        ,colsample_bytree = 0.9))
        mdl.append(RandomForestClassifier(n_estimators=20,max_depth=10,
                                            random_state=0,class_weight=None,
                                            n_jobs=-1))
        mdl.append(ExtraTreesClassifier())
        ml1=['XGBoost','Random_Forest','Extra_Tree']
        feat_sel=[]
        for i in range(3):

            model=mdl[i]
            model.fit(X_train, y_train)
            model.feature_importances_
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feat_labels = X_train.columns
            print("Feature ranking:") 
            sel_feat=[]
            for f in range(X_train.shape[1]):
                    print("%d. feature no:%d feature name:%s (%f)" % (f+1, indices[f], feat_labels[indices[f]], importances[indices[f]]))
                    sel_feat.append(feat_labels[indices[f]])
            top_n=15
            feat_sel.append(sel_feat)
            indices = indices[0:top_n]
            plt.subplots(figsize=(12, 10))
            g = sns.barplot(importances[indices],feat_labels[indices], orient='h',label='big') #import_feature.iloc[:Num_f]['col'].values[indices]

            g.set_title(ml1[i]+' feature selection',fontsize=25)
            g.set_xlabel("Relative importance",fontsize=25)
            g.set_ylabel("Features",fontsize=25)
            g.tick_params(labelsize=14)
            sns.despine() 
                # plt.savefig('feature_importances_v3.png')
            plt.show()
            print('-----------------------------------------------------------------')
        xgboost=feat_sel[0]
        randomforest=feat_sel[1]
        extratree=feat_sel[2]
        return(xgboost,randomforest,extratree)

