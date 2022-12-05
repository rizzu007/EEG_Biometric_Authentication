from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
# !pip install eeglib
from scipy.signal import butter, lfilter

from scipy import signal

def segmentation(x, time_window):
    seg_data = []
    for i in range(0, x.shape[0] - time_window + 1, time_window):
        seg_data.append(x[i:i+time_window])
    
    return np.array(seg_data)

from sklearn.decomposition import PCA
def pca_(xtr,xts,n_components=20):
    pca = PCA(n_components=20, whiten=True)
    pca.fit(xtr)
    trainx = pca.transform(xtr)
    testx = pca.transform(xts)
    return trainx,testx

def resample_input(input_signal,wd_size):

    z1 = (input_signal.shape[0])/wd_size
    z1 = int(z1)
    all_signal = np.zeros((z1, wd_size, 14))

    for ch in range(14):
        ch_signals = input_signal[:,ch]
        s = np.array([])
        for signal_segment in ch_signals:
            
            #signal_resampled = signal.resample(signal_segment, 128)
            s = np.hstack((s, signal_segment))

        segmented_s = segmentation(s, wd_size)
        #print(segmented_s.shape)

        all_signal[:,:,ch] = segmented_s

    return all_signal



def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y




### Feature computation

def mean(data):
    return np.mean(data,axis=-1)
    
def std(data):
    return np.std(data,axis=-1)

def ptp(data):
    return np.ptp(data,axis=-1)

def var(data):
        return np.var(data,axis=-1)

def minim(data):
      return np.min(data,axis=-1)


def maxim(data):
      return np.max(data,axis=-1)

def argminim(data):
      return np.argmin(data,axis=-1)


def argmaxim(data):
      return np.argmax(data,axis=-1)

def mean_square(data):
      return np.mean(data**2,axis=-1)

def rms(data): #root mean square
      return  np.sqrt(np.mean(data**2,axis=-1))  

def abs_diffs_signal(data):
    return np.sum(np.abs(np.diff(data,axis=-1)),axis=-1)


def skewness(data):
    return stats.skew(data,axis=-1)

def kurtosis(data):
    return stats.kurtosis(data,axis=-1)

def concatenate_features1(data):
    return np.concatenate((np.array(mean(data)).reshape(1),np.array(std(data)).reshape(1),np.array(ptp(data)).reshape(1),
                    np.array(var(data)).reshape(1),np.array(minim(data)).reshape(1),np.array(maxim(data)).reshape(1),np.array(argminim(data)).reshape(1),np.array(argmaxim(data)).reshape(1),
                    np.array(mean_square(data)).reshape(1),np.array(rms(data)).reshape(1),np.array(abs_diffs_signal(data)).reshape(1),np.array(skewness(data)).reshape(1),
                    np.array(kurtosis(data)).reshape(1)),axis=0)


from eeglib import wrapper, helpers
def frq_feat(ss):
  helper = helpers.Helper(ss)

  wrap = wrapper.Wrapper(helper)
  wrap.addFeature.PFD()
  wrap.addFeature.bandPower()
  wrap.addFeature.hjorthActivity()
  wrap.addFeature.hjorthMobility()
  wrap.addFeature.hjorthComplexity()
  wrap.addFeature.LZC()
  wrap.addFeature.sampEn()
  wrap.addFeature.DFA()
  wrap.addFeature.HFD()
  # wrap.addFeature.DFT()
  wrap.addFeature.synchronizationLikelihood()

  features=wrap.getAllFeatures()
  return np.array(features)