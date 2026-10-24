#Imports
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
scores = []
result_list = []
cv = ShuffleSplit(test_size=0.2,random_state=67)
lda = LDA()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
clf = Pipeline([("CSP",csp),("LDA",lda)])
for i in range(1,18):
    reali = i
    if i >= 8:
        i+=1
    if i >= 13:
        i+=1
    if i >= 15:
        i+=1
    #Load Data
    if i < 10:
        PATHS = [f"Data/Training Set/S00{i}-001_eeg.csv", f"Data/Training Set/S00{i}-002_eeg.csv", f"Data/Training Set/S00{i}-003_eeg.csv"]
    else:
        PATHS = [f"Data/Training Set/S0{i}-001_eeg.csv", f"Data/Training Set/S0{i}-002_eeg.csv", f"Data/Training Set/S0{i}-003_eeg.csv"]
    if i == 7:
        PATHS.remove(f"Data/Training Set/S007-002_eeg.csv") #Remove the corrupted file
    #Declare Constants
    SFREQ = 250 #Frequency of the eeg
    ch_names = ['C4', 'C3', 'Cz','Fz','PO7','Pz','PO8','Oz'] #Channels used in the eeg

    #Converting csv to rawArray for mne
    df = pd.concat([pd.read_csv(path) for path in PATHS],ignore_index=True)

    #Data
    eegdata = np.concatenate([pd.read_csv(path)[ch_names].to_numpy().T for path in PATHS],axis=1)

    #Information
    info = mne.create_info(ch_names=ch_names,sfreq=SFREQ,ch_types="eeg")

    #Converting marker to annotation
    markers =df["Marker_val"]
    cue_indices = df.index[markers.isin(['move', 'rest'])].tolist() #Get index of when move and rest is happening
    onset = []
    duration = []
    description = []
    for label_idx in cue_indices:
        label_str = markers.loc[label_idx]
        start_second = label_idx/SFREQ
        onset.append(start_second)
        description.append(label_str)
    anno = mne.Annotations(onset=onset,duration=0,description=description) 

    raw = mne.io.RawArray(eegdata,info)
    raw.set_annotations(anno)
    raw.set_montage("colin27_1005")
    raw.filter(l_freq=7.,h_freq=30.)
    epochs = mne.Epochs(
        raw,
        event_id=["move","rest"],
        tmin = 0.5,
        tmax = 4.5,
        proj=True,
        baseline=None,
        preload=True,
    )        
    epochs_train = epochs.copy()
    label = epochs.events[:,2]
    epochs_data = epochs.get_data(copy = False)
    epochs_data_train = epochs_train.get_data(copy = False)
    
    if i < 10:
        PATH = f"Data/Single subject test set/S00{i}_test.csv"
    else:
        PATH = f"Data/Single subject test set/S0{i}_test.csv"

    #Converting csv to rawArray for mne
    df = pd.read_csv(PATH)

    #Data
    eegdata = pd.read_csv(PATH)[ch_names].to_numpy().T

    #Information
    info = mne.create_info(ch_names=ch_names,sfreq=SFREQ,ch_types="eeg")

    #Converting marker to annotation
    markers =df["Marker_val"]
    cue_indices = df.index[markers.isin(["cue_start"])].tolist() #Get index of when move and rest is happening
    onset = []
    duration = []
    description = []
    for label_idx in cue_indices:
        label_str = markers.loc[label_idx]
        start_second = label_idx/SFREQ
        onset.append(start_second)
        description.append(label_str)
    anno = mne.Annotations(onset=onset,duration=0,description=description) 

    raw = mne.io.RawArray(eegdata,info)
    raw.set_annotations(anno)
    raw.set_montage("colin27_1005")
    raw.filter(l_freq=7.,h_freq=30.)
    epochs = mne.Epochs(
        raw,
        event_id=["cue_start"],
        tmin = 0.5,
        tmax = 4.5,
        proj=True,
        baseline=None,
        preload=True,
    )        
    epochs_test = epochs.copy()
    clf.fit(epochs_data_train, label)
    results = clf.predict(epochs_test.get_data(copy = False))
    binah = ["move","rest"]
    dataframe = []
    scores.append(np.mean(cross_val_score(clf, epochs_data_train, label, cv=cv)))
    for b in range(len(results)):
        dataframe.append({"ID":(reali-1)*len(results)+b, "TARGET": binah[results[b]-1]})
    result_list.append(pd.DataFrame(dataframe))
result = pd.concat(result_list, ignore_index=True)
result.to_csv("results.csv", index=False)
print(f"Classification accuracy: {np.mean(scores)}")