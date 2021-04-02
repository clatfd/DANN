from sklearn import metrics
import numpy as np

def getauc(pred, y):
    # pred -1,0,1 continuous
    # y 1,2,3 categorical
    auc_adv = np.nan
    auc_all = np.nan

    ylesion = []
    yadv = []
    for yi in y:
        if yi >= 2:
            ylesion.append(2)
        else:
            ylesion.append(1)
        if yi >= 3:
            yadv.append(2)
        else:
            yadv.append(1)
    if 1 in yadv and 2 in yadv:
        fpr, tpr, thresholds = metrics.roc_curve(np.array(yadv), np.array(pred), pos_label=2)
        auc_adv = metrics.auc(fpr, tpr)

    if 1 in ylesion and 2 in ylesion:
        fpr, tpr, thresholds = metrics.roc_curve(np.array(ylesion), np.array(pred), pos_label=2)
        auc_all = metrics.auc(fpr, tpr)
    return auc_adv, auc_all


def decidelesion(predbatch,thres_all=-0.5,thres_adv=0.05):
    assert thres_all<=thres_adv
    #chan 0: label, 1: confidence
    #for mi in range(predbatch.shape[0]):
    if predbatch<thres_all:
        labelbatch = 0
        labelconf = max(0.01,1-abs(predbatch-(-1)))
    elif predbatch>thres_adv:
        labelbatch = 2
        labelconf = max(0.01,1-abs(predbatch-2))
    else:
        labelbatch = 1
        labelconf = max(0.01,1-abs(predbatch))
    return labelbatch,labelconf
