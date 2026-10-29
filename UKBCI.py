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
#BAND = [[(4,6)],[(11,13),(20,22)],[(21,23)],[(4,6),(13,16)],[(10,12)],[(7,8)],[(24,27)],[],[(10,12)],[(4,6),(10,11)],[(12,13)],[(9,11),(20,21)],[],[(9,11),(18,20)],[],[(5,7)],[(2,4),(7,9)],[(6,7),(13,14),(26,27)],[(27,29),(10,14)],[(12,13)]]
BAND = [
    [(4, 9)],                  # S1  - Claude
    [(11, 13), (20, 22)],      # S2  - You
    [(21, 23)],                # S3  - You
    [(4, 7), (12, 18)],        # S4  - Claude
    [(10, 12)],                # S5  - You
    [(7, 8)],                  # S6  - You
    [(24, 27)],                # S7  - You
    [],                        # S8  - skipped
    [(10, 12)],                # S9  - You
    [(3, 8), (10, 12)],        # S10 - Claude
    [(12, 13)],                # S11 - You
    [(9, 11), (20, 21)],       # S12 - You
    [],                        # S13 - skipped
    [(8, 13), (17, 22)],       # S14 - Claude
    [],                        # S15 - skipped
    [(5, 7)],                  # S16 - You
    [(2, 4), (7, 9)],          # S17 - You
    [(5, 18), (26, 29)],       # S18 - Claude
    [(9, 17), (26, 30)],       # S19 - Claude
    [(12, 13)],                # S20 - You
]

WEIGHT = [
    [1.00],                    # S1
    [0.50, 0.50],              # S2
    [1.00],                    # S3
    [0.54, 0.46],              # S4
    [1.00],                    # S5
    [1.00],                    # S6
    [1.00],                    # S7
    [],                        # S8
    [1.00],                    # S9
    [0.53, 0.47],              # S10
    [1.00],                    # S11
    [0.50, 0.50],              # S12
    [],                        # S13
    [0.52, 0.48],              # S14
    [],                        # S15
    [1.00],                    # S16
    [0.50, 0.50],              # S17
    [0.49, 0.51],              # S18
    [0.44, 0.56],              # S19
    [1.00],                    # S20
]
#bandpass = [(i,i+1) for i in range(1,30)]
cv = ShuffleSplit(test_size=0.2,random_state=67)
lda = LDA()
csp = CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False)
clf = Pipeline([("CSP",csp),("LDA",lda)])
mark = [["move","rest"],["cue_start"]]

def convertRaw(PATH,target):
    #0-> train 1->test
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
        epochs.append(epoch)
    labels = epochs[0].events[:,2]
    return(epochs,labels)

for subject in range(1,21):
    bandpass = BAND[subject-1]
    weight = WEIGHT[subject-1]
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
    train_list = [] 
    labels = []
    for i in range(3):
        try:
            e, l = convertRaw(train_path[i],0)
        except:
            continue
        train_list.append(e)
        labels.extend(l)
    epoch_train = []
    for i in range(len(bandpass)):
        epoch_train.append(mne.concatenate_epochs([train[i] for train in train_list]))
    epoch_test = convertRaw(test_path,1)[0]
    prob = []
    score = []
    for i in range(len(bandpass)):
        scores = cross_val_score(clf, epoch_train[i].get_data(), labels, cv=cv)
        score.append(np.mean(scores))
        clf.fit(epoch_train[i].get_data(),labels)
        prob.append(clf.predict_proba(epoch_test[i].get_data())*weight[i])
    pred = np.argmax(np.array(prob).mean(0),axis=1)
    target = np.array(["move", "rest"])[pred]
    score_list.append(score)
    result_list.extend(target)
print(score_list)
dataframe = []
for i in range(len(result_list)):
    dataframe.append({"ID":i,"TARGET":result_list[i]})
result_df = pd.DataFrame(dataframe)
result_df.to_csv("results.csv",index=False)
