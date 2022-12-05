
from utils import*

def data_processing(feat_file,window_size):
    dt=feat_file
    # window_size=128
    sub=list(dt.keys())
    # sub=['s21_s2']
    # all_feat=[]
    all_lbl=[]
    all_sub=[]
    for a,k in enumerate(sub):
        tt=np.array(dt[k]['Data'])
        lb=np.array(dt[k]['Label'])
        features1=[]
        label1=[]
        subj=[]
        # tt=resample_input(tt)
        for j in range(tt.shape[0]):
            
            tt1=tt[j]
            try: 
                tt1=resample_input(tt1,window_size) ## resampling with window size 256
                # print(tt1.shape)
            except:
                pass
            for l in range(tt1.shape[0]):
                # lb1=lb[j]
                
                try:
                  features=[]
                    
                  for i in range(tt1.shape[2]):
                      sg=tt1[l,:,i]                       
                      sg = butter_bandpass_filter(sg, 0.6, 50, 256, order=3)
                      fea1=concatenate_features1(sg)
                      sg1=sg.reshape(1,sg.shape[0])
                      fea2=frq_feat(sg1)
                      fea2=fea2.reshape(fea2.shape[1])
                      fea11=np.concatenate((fea1,fea2),0)
                      features.extend(fea11)
                      lb1=lb[j]
                      sub1=k
                  features=np.array(features)
                  features1.append(features)
                  label1.append(lb1)
                  subj.append(sub1)
                except:
                #         # print(k)
                        pass
                        # print(tt1.shape)
                  

            
        features1=np.array(features1)
        label1=np.array(label1)
        # print(features1.shape)
        try:
            
            if a==0:
                all_feat=features1
            else:
                all_feat=np.concatenate((all_feat,features1),axis=0)
            all_lbl.extend(label1)
            all_sub.extend(subj)
        except:
            pass
        
    cnl=['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
    feat=['mean','std','ptp','var','minim','maxim','argminim','argmaxim','mean_square','rms','abs_diffs_signal','skewness','kurtosis',
          'PFD(){}_0', 'bandPower_delta', 'bandPower_theta',
        'bandPower_alpha', 'bandPower_beta', 'hjorthActivity',
        'hjorthMobility', 'hjorthComplexity', 'LZC',
        'sampEn', 'DFA', 'HFD'
          ]

    col=[]

    for i in cnl:
        for j in feat:
            col.append(f'{i}_{j}')
    fd=pd.concat([pd.DataFrame(all_feat,columns=col),pd.DataFrame(all_sub,columns=['sub_session']),pd.DataFrame(all_lbl,columns=['Label'])],1)
    fd=fd.replace('SSVEPC_SSVEPC', 'SSVEPC_5Hz')
    fd=fd.replace('SSVEP_SSVEP', 'SSVEP_5Hz')
    fd=fd.replace('EYES_EYES', 'EYES_OPEN')
    fd.to_csv(r'Final_full_data1n.csv')
