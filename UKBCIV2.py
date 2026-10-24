#Imports
import mne
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, ShuffleSplit

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

mne.set_log_level("WARNING")
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Variable & Constant
result_list = []
score_list = []
SFREQ = 250 #Frequency of the eeg
ch_names = ['C4', 'C3', 'Cz','Fz','PO7','Pz','PO8','Oz'] #Channels used in the eeg
#bandpass = [(i,i+1) for i in range(1,30)]
bandpass = [(1,7),(7,30)]
cv = ShuffleSplit(test_size=0.2,random_state=67)
lda = LDA()
csp = CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False)
clf = Pipeline([("CSP",csp),("LDA",lda)])
mark = [["move","rest"],["cue_start"]]

def convertRaw(PATH,target):
    #0-> train 1->test
    if target == 0:
        df = pd.concat([pd.read_csv(path) for path in PATH],ignore_index=True)
    else:
        df = pd.read_csv(PATH)
    data = df[ch_names].to_numpy().T
    info = mne.create_info(ch_names=ch_names,sfreq=SFREQ,ch_types="eeg")
    #Converting marker to annotation
    markers =df["Marker_val"]
    cue_indices = df.index[markers.isin(mark[target])].tolist() #Get index of when move and rest is happening
    onset = []
    duration = []
    description = []
    for label_idx in cue_indices:
        label_str = markers.loc[label_idx]
        start_second = label_idx/SFREQ
        onset.append(start_second)
        description.append(label_str)
    anno = mne.Annotations(onset=onset,duration=0,description=description) 
    raw = mne.io.RawArray(data,info)
    raw.set_annotations(anno)
    raw.set_montage("colin27_1005")
    epochs = []
    labels = []
    for low,high in bandpass:
        raw_band = raw.copy().filter(l_freq=low,h_freq=high)
        epoch = mne.Epochs(
            raw_band,
            event_id=mark[target],
            tmin = 0.5,
            tmax = 4.5,
            proj=True,
            baseline=None,
            preload=True,
        )
        label = epoch.events[:,2]
        epochs.append(epoch)
        labels.append(label)
    if target == 0:
        return(epochs,labels)
    else:
        return(epochs)

for subject in range(1,21):
    train_path = []
    for i in range(1,4):
        if os.path.exists(f"Data/Training Set/S{"0"*(2-int(math.log(subject,10)))+str(subject)}-00{i}_eeg.csv"):
            train_path.append(f"Data/Training Set/S{"0"*(2-int(math.log(subject,10)))+str(subject)}-00{i}_eeg.csv")
        else:
            print(f"Subject {subject} trial {i} missing")
    if len(train_path) != 0:
        test_path = f"Data/Single subject test set/S{"0"*(2-int(math.log(subject,10)))+str(subject)}_test.csv"
    else:
        print("No files")
        continue
    epoch_train, labels = convertRaw(train_path,0)
    epoch_test = convertRaw(test_path,1)
    prob = []
    score = []
    for i in range(len(bandpass)):
        scores = cross_val_score(clf, epoch_train[i].get_data(), labels[i], cv=cv)
        score.append(np.mean(scores))
        clf.fit(epoch_train[i].get_data(),labels[i])
        prob.append(clf.predict_proba(epoch_test[i].get_data()))
    pred = np.argmax(np.array(prob).mean(0),axis=1)
    target = np.array(["move", "rest"])[pred]
    score_list.append(score)
    result_list.extend(target)
print(np.mean(score_list,axis=0))
dataframe = []
for i in range(len(target)):
    dataframe.append({"ID":i,"TARGET":target[i]})
result_df = pd.DataFrame(dataframe)
