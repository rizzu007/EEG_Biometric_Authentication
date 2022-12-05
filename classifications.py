from library import*
from models import*

def classification(data,classifier,feature_selection_model,feature_num,feat_increment,ind):

    c=classifier
    xtrain,xtest,ytrain,ytest=data['data']
    num_feat=feature_num
    fsm=feature_selection_model
    feature=fsm[0:num_feat]
    clf,clff=models()
    clf1=clf[c]
    acc=[]
    prc=[]
    sns=[]
    spc=[]
    f1s=[]
    for i in list(range(0,num_feat,feat_increment)):

                    y_pred=[]
                    y2=[]
                    tl=fsm[0:i+1]  #feature increasing
                
                    for k in range(1):
                        x11=pd.DataFrame(xtrain[k])
                        x11.columns=ind
                        x1=x11[tl]
                        y1=ytrain[k]   
                        model = clf1.fit(np.array(x1),np.array(y1))
                        #model = clf1.fit(x[train],y.iloc[train])
                        xts=pd.DataFrame(xtest[k])
                        xts.columns=ind
                        xt1=xts[tl]
                        y_pr=model.predict(xt1)
                        y_pred.extend(y_pr)
                        y2.extend(ytest[k])
                        prediction=model.predict_proba(np.array(xt1))

                    y21=y2
                    y_pred1=y_pred
                    categories=list(pd.Series(y2).unique())
                    ac = accuracy_score(y2, y_pred)
                    #y_pred = cross_val_predict(clf, X, y, cv=5)
                    pr = precision_score(y2, y_pred, average='weighted')
                    rc = recall_score(y2, y_pred, average='weighted')
                    sp = recall_score(y2, y_pred, average='weighted',pos_label=1)
                    f1 = f1_score(y2, y_pred, average='weighted')   

                    from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
                    # main confusion matrix
                    cm = confusion_matrix(y21, y_pred1)
                    cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
                    # print(cm_per_class)
                    # Overall Accuracy
                    Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
                    Overall_Accuracy = round(Overall_Accuracy*100, 2)
                    # create confusion matrix table (pd.DataFrame)
                    # cm_table = pd.DataFrame(cm, index=categories , columns=categories)
                    if (i+1)!=1:
                      feature_no= 'top_'+str(i+1)+'_features'
                    else:
                      feature_no= 'top_'+str(i+1)+'_feature'
                    # print(ac)
                    acc.append(round(ac*100,2))
                    prc.append(round(pr*100,2))
                    spc.append(round(sp*100,2))
                    sns.append(round(rc*100,2))
                    f1s.append(round(f1*100,2))

    Result=pd.concat([pd.DataFrame(acc),pd.DataFrame(sns),pd.DataFrame(prc),pd.DataFrame(spc),pd.DataFrame(f1s)],1)
    Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score']
    # Result.index= feat
    #Result.to_csv
    print('---------------------------------------------------------------------')
    print('Result for '+clff[c]+' classifier')
    print('---------------------------------------------------------------------')
    return Result,y2,y_pred


from library import*
from models import*

def classification_pca(data,classifier,ind):
    c=classifier
    xtrain,xtest,ytrain,ytest=data['data']

    # ind=data['index'].to_list()

    # num_feat=feature_num
    # fsm=feature_selection_model
    # feature=fsm[0:num_feat]
    clf,clff=models()
    clf1=clf[c]
    acc=[]
    prc=[]
    sns=[]
    spc=[]
    f1s=[]
    for i in list(range(1)):

                    y_pred=[]
                    y2=[]            
                    for k in range(1):
                        x1=pd.DataFrame(xtrain[k])
                        # x11.columns=ind
                        # x1=x11[tl]
                        y1=ytrain[k]   
                        model = clf1.fit(np.array(x1),np.array(y1))
                        #model = clf1.fit(x[train],y.iloc[train])
                        xt1=pd.DataFrame(xtest[k])

                        y_pr=model.predict(xt1)
                        y_pred.extend(y_pr)
                        y2.extend(ytest[k])
                        prediction=model.predict_proba(np.array(xt1))

                    y21=y2
                    y_pred1=y_pred
                    categories=list(pd.Series(y2).unique())
                    ac = accuracy_score(y2, y_pred)
                    #y_pred = cross_val_predict(clf, X, y, cv=5)
                    pr = precision_score(y2, y_pred, average='weighted')
                    rc = recall_score(y2, y_pred, average='weighted')
                    sp = recall_score(y2, y_pred, average='weighted',pos_label=1)
                    f1 = f1_score(y2, y_pred, average='weighted')   

                    from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
                    # main confusion matrix
                    cm = confusion_matrix(y21, y_pred1)
                    cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
                    # print(cm_per_class)
                    # Overall Accuracy
                    Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
                    Overall_Accuracy = round(Overall_Accuracy*100, 2)
                    # create confusion matrix table (pd.DataFrame)
                    # cm_table = pd.DataFrame(cm, index=categories , columns=categories)
                    if (i+1)!=1:
                      feature_no= 'top_'+str(i+1)+'_features'
                    else:
                      feature_no= 'top_'+str(i+1)+'_feature'
                    # print(ac)
                    acc.append(round(ac*100,2))
                    prc.append(round(pr*100,2))
                    spc.append(round(sp*100,2))
                    sns.append(round(rc*100,2))
                    f1s.append(round(f1*100,2))
    Result=pd.concat([pd.DataFrame(acc),pd.DataFrame(sns),pd.DataFrame(prc),pd.DataFrame(spc),pd.DataFrame(f1s)],1)
    Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score']
    # Result.index= feat
    #Result.to_csv
    print('---------------------------------------------------------------------')
    print('Result for '+clff[c]+' classifier')
    print('---------------------------------------------------------------------')
    return Result,y2,y_pred
