import os
import pickle

def data_segmentation(file_dir):
    dataf={}
    fl=os.listdir(file_dir)
    files1=[x.split('.')[0] for x in fl]

    for sub_session in files1:

            sub_session1=sub_session+'.mat'
            mat_file=os.path.join(file_dir,sub_session)

            mat=scipy.io.loadmat(mat_file)
            eve=pd.DataFrame(mat['events'])
            start=[]
            end=[]
            cls=[]
            cls_hz=[]
            for i in range(eve.shape[0]):
                start.append(eve[0][i][0][0])
                end.append(eve[1][i][0][0])
                cls.append(eve[2][i]['STIMULI'][0][0][0])
        # print(eve[2][i][0][0][0][0])
                try:
                    cls_hz.append(eve[2][i][0][0][0][0])
                except:
                    cls_hz.append('None')
            f_cls=[]
            for j in range(len(cls)):
                a=cls[j]
                b=cls_hz[j]
                if cls[j]=='SSVEPC' or cls[j]=='SSVEP' or cls[j]=='EYES':
                    f_cls.append(f'{a}_{b}')
                else:
                    f_cls.append(cls[j])



            rec=pd.DataFrame(mat['recording'])
            rec.columns=['counter', 'interpolated', 'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4','timestamp']
            cnl=['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']


            rec1=rec.iloc[:,2:-1]
            rec2=[round(x,1) for x in list(rec1.index)]
            data=[]
            er_ind=[]
            lbl=[]
            for ind in range(eve.shape[0]):
                try:
                    rec2=[round(x,1) for x in list(rec.timestamp)]
                    a,b=rec2.index(round(start[ind],1)),rec2.index(round(end[ind],1))
                    dat=np.array(rec1.iloc[a:b,:])
                    data.append(dat)
                    lbl.append(f_cls[ind])
                except:
                    er_ind.append(ind)
                    # print('not find in the list')

            for j in er_ind: 
                try:
                    rec2=[round(x) for x in list(rec.timestamp)]
                    a,b=rec2.index(round(start[j]+1)),rec2.index(round(end[j]-1))
                    dat=np.array(rec1.iloc[a:b,:])
                    data.append(dat)
                    lbl.append(f_cls[j])  
                except:
                   # print('not find in the list')
                   pass
            # dataf.append()
            dd={sub_session: {'Data':data,'Label':lbl}}

            dataf.update(dd)

    pickle.dump(dataf,open('full_data1.pkl','wb'))


