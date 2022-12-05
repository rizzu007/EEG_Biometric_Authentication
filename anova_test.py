from scipy.stats import f_oneway
from library import*
def anova_test(data,feature_selection):

    data['subject']=pd.DataFrame([x.split('_')[0] for x in data.sub_session])
    x_tr1=data.iloc[:,2:-3]
    y_tr1=data['Label']
    lbl=list(y_tr1.unique())
    #col=list(data.columns)[2:15]
    col = feature_selection[:30]
    stt=[]
    pp=[]
    dst=[]

    for j,k in enumerate(col):
      for  i in lbl:
        if i=='IMAGE':
          da1=data[data['Label']==i]
          grp1=list(da1[col[j]])
        elif i=='COGNITIVE':
          da1=data[data['Label']==i]
          grp2=list(da1[col[j]])
        elif i=='SSVEPC_5Hz':
          da1=data[data['Label']==i]
          grp3=list(da1[col[j]])
        elif i=='SSVEP_5Hz':
          da1=data[data['Label']==i]
          grp4=list(da1[col[j]])
        elif i=='REST':
          da1=data[data['Label']==i]
          grp5=list(da1[col[j]])
        elif i=='EYES_OPEN':
          da1=data[data['Label']==i]
          grp6=list(da1[col[j]])
        elif i=='SSVEPC_3Hz':
          da1=data[data['Label']==i]
          grp7=list(da1[col[j]])
        elif i=='SSVEPC_7Hz':
          da1=data[data['Label']==i]
          grp8=list(da1[col[j]])
        elif i=='SSVEPC_10Hz':
          da1=data[data['Label']==i]
          grp9=list(da1[col[j]])
        elif i=='SSVEP_3Hz':
          da1=data[data['Label']==i]
          grp10=list(da1[col[j]])
        elif i=='SSVEP_7Hz':
          da1=data[data['Label']==i]
          grp11=list(da1[col[j]])
        elif i=='SSVEP_10Hz':
          da1=data[data['Label']==i]
          grp12=list(da1[col[j]])
        elif i=='EYES_CLOSED':
          da1=data[data['Label']==i]
          grp13=list(da1[col[j]])
      stat, p = f_oneway(grp1, grp2, grp3,grp4,grp5,grp6,grp7,grp8,grp9,grp10,grp11,grp12,grp13)
      if p > 0.05:
        dst.append('same')
        # print('Probably the same distribution')
      else:
        dst.append('different')
        # print('Probably different distributions')
      stt.append(stat)
      pp.append(round(p,6))
    stt=pd.DataFrame(stt,columns=['stat score'])
    pp=pd.DataFrame(pp,columns=['p-values'])
    dst=pd.DataFrame(dst,columns=['distribution'])
    st_analysis=pd.concat([stt,pp,dst],axis=1)
    st_analysis.index=col
    print(st_analysis)    
        