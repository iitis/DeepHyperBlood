# -*- coding: utf-8 -*-
"""
************************************************************************
Copyright 2020 Institute of Theoretical and Applied Informatics, 
Polish Academy of Sciences (ITAI PAS) https://www.iitis.pl
author: K. Książek, P.Głomb, M. Romaszewski

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
************************************************************************

Code for experiments in the paper by  
K. Książek, M. Romaszewski, P. Głomb, B. Grabowski, M. Cholewa
`Blood Stains Classification with Hyperspectral
Imaging and Deep Neural Networks'

SVM experiments
"""


import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from data_library import get_data,get_anno
from sklearn.model_selection import StratifiedKFold
from two_stage_svm import TwoStageSVM
from sklearn.metrics._classification import accuracy_score
from sklearn.preprocessing import scale
from os.path import isfile
from sklearn.metrics import cohen_kappa_score
from sklearn.decomposition import PCA

from data_loader import blood_loader
from data_paths import PATH_DATA,PATH_ANNO

DS = ['E_1', 'E_7', 'E_21', 'F_1', 'F_1a', 'F_2', 'F_2k', 'F_7', 'F_21']
DS_I = [('F_1', 'E_1'), ('F_1a', 'E_1'), ('F_2', 'E_7'), ('F_2k', 'E_7')
    ,('F_7', 'E_7'), ('F_21', 'E_21'), ('F_2', 'F_2k'), ('F_2k', 'F_2')]

#************************** UTILITIES  *******************************************

def calculate_average_accuracy(ground_truth,
                               prediction):
    '''
    Calculate accuracy for each class in ground truth
    (without background) and average accuracy

    Arguments:
        ground_truth: map of ground truth of a given image
        prediction: map of prediction of an algorithm

    Returns:
        classes_accuracies: a list of accuracies for each class
                            without background
        average_accuracy: average accuracy (mean from consecutive classes)
    '''
    # all classes without background
    classes = np.unique(ground_truth)
    classes = np.delete(classes, np.where(classes == 0))

    classes_accuracies = []
    # for each class
    for no in classes:
        class_occurences = np.argwhere(ground_truth == no)
        # calculate total number of pixels from a given class
        total_occurences = class_occurences.shape[0]
        correct_occurences = 0
        for occurence in class_occurences:
            if(prediction[tuple(occurence)] == no):
                correct_occurences += 1
        class_accuracy = (correct_occurences / total_occurences)
        classes_accuracies.append(class_accuracy)
    # average accuracy
    average_accuracy = np.mean(classes_accuracies)

    return average_accuracy


def calculate_kappa_coefficient(ground_truth,
                                prediction):
    '''
    Kamil's function
    Arguments:
        ground_truth: map of ground truth of a given image (2D)
        prediction: map of prediction of an algorithm (2D)

    Returns:
        kappa: value of Cohen's kappa coefficient
    '''
    # Prepare a proper form of results
    ground_truth = ground_truth.reshape(-1)
    prediction = prediction.reshape(-1)

    # Remove background indices
    background = np.argwhere(ground_truth == 0)
    ground_truth = np.delete(ground_truth, background)
    prediction = np.delete(prediction, background)

    # Calculate Cohen's kappa value
    kappa = cohen_kappa_score(ground_truth, prediction)
    return kappa 

def spectra_normalisation(X,mode='median'):
    """
    performs spectra normalisation, dividing each spectrum by their median/mean
    
    parameters:
    X: 2D data array
    mode: per-spectra normalisation[median,mean,max]

    returns:
    2D array copy with normalised spectra
    """
    X2 = X.copy()
    norms = None
    if mode == 'median': 
        norms = np.median(X2,axis=1)
    elif mode == 'mean':     
        norms = np.mean(X2,axis=1)
    elif mode == 'max':     
        norms = np.max(X2,axis=1)
    else:
        raise NotImplementedError    
    for i in range(len(norms)):
        X2[i]=X2[i]/norms[i]
    return X2

def preprocess(X_in,preprocessing='none'):
    """
    perform spectra normalisation and applies selected preprocessing e.g. scaling

    parameters:
    X_in: 2D data array
    preprocessing: type of prepoecessing ['none','scale']

    returns:
    2D array copy with preprocessed and normalised spectra    
    """
    X=X_in.copy()
    X = spectra_normalisation(X)
    if preprocessing == 'scale':
        X=scale(X)
    elif preprocessing == 'none':
        pass
    else:
        raise NotImplementedError    
    return X 

def load_ds(name='F(1)',exp_index=0):
    """
    loads a dataset image with annotation

    parameters:
    name: image name
    exp_index: no. experiment (e.g. 0-9)

    returns:
    data,train_gt,test_gt,anno

    """
    data,_ = get_data(name)
    anno = get_anno(name)
    train_gt = np.load("{}{}/train_gt_{}.npz".format(PATH_ANNO,name,exp_index))['gt']
    test_gt = np.load("{}{}/test_gt_{}.npz".format(PATH_ANNO,name,exp_index))['gt']
    return data,train_gt,test_gt,anno


#************************** EXPERIMENTS  *******************************************

def experiment_transductive(name='E(1)',exp_index=0,preprocessing='none'):
    """
    performs transductive experiment for frame/target pair

    parameters:
    name: image name
    exp_index: no. experiment (e.g. 0-9)
    preprocessing: preprocessing mode

    returns:
    True/False
    """
    start = time.time()
    outfile = 'results/{}_{}_{}.npz'.format(name,exp_index,preprocessing)
    data,train_gt,test_gt,_ = load_ds(name,exp_index)

    #preprocessing
    X = data.reshape(-1,data.shape[2])
    X=preprocess(X,preprocessing=preprocessing)
    data = X.reshape(data.shape)
     
    X_train = data[train_gt>0]
    y_train = train_gt[train_gt>0]
    time_prepare = time.time()-start

    start = time.time()
    svm=TwoStageSVM()
    svm.fit(X_train,y_train)
    time_train = time.time()-start

    X_test = data[test_gt>0]
    y_test = test_gt[test_gt>0]
    
    start = time.time()
    y_pred = svm.predict(X_test) 
    time_test = time.time()-start
    y_pred_all = svm.predict(X)
    
    acc = accuracy_score(y_test,y_pred)
    np.savez_compressed(outfile,y_test=y_test,y_pred=y_pred,sh=data.shape,best_params=[svm.params_['C'],svm.params_['gamma']],acc=[acc],cls_score=[svm.score_],y_pred_all=y_pred_all,time_train=time_train,time_test=time_test,time_prepare=time_prepare)
    print (outfile,acc,time_train,time_test)




def experiment_inductive(name_train='hyperblood_frame_day_1_afternoon',name_test='hyperblood_comparison_day_1',exp_index=0,preprocessing='none'):
    """
    performs inductive experiment for frame/target pair

    parameters:
    name_train: image name (for training)
    name_test: image name (for testing)
    exp_index: no. experiment (e.g. 0-9)
    preprocessing: preprocessing mode
    """    
    print (name_train,name_test)
    start = time.time()
    outfile = 'results_i/{}_{}_{}_{}.npz'.format(name_train,name_test,exp_index,preprocessing)

    data_train,train_gt,_,_ = load_ds(name_train,exp_index)
    data_test,_,test_gt,_ = load_ds(name_test,exp_index)
    
    _,anno_test, _,_,_ = blood_loader(PATH_DATA, name_test)
    
    test_gt = np.asarray(test_gt,dtype=np.int32)
    anno_test = np.asarray(anno_test,dtype=np.int32)
    assert np.sum(np.unique(anno_test)-np.unique(test_gt))==0
    
    #preprocessing
    X = data_train.reshape(-1,data_train.shape[2])
    X=preprocess(X,preprocessing=preprocessing)
    data_train = X.reshape(data_train.shape)

    X = data_test.reshape(-1,data_test.shape[2])
    X=preprocess(X,preprocessing=preprocessing)
    data_test = X.reshape(data_test.shape)

    X_train = data_train[train_gt>0]
    y_train = train_gt[train_gt>0]
    time_prepare = time.time()-start

    start = time.time()
    svm=TwoStageSVM()
    svm.fit(X_train,y_train)
    time_train = time.time()-start

    #second version of results
    X_test_gt = data_test.copy()[anno_test>0]
    y_test_gt = anno_test.copy()[anno_test>0]

    X_test = data_test[test_gt>0]
    y_test = test_gt[test_gt>0]

    
    y_pred = svm.predict(X_test) 
    y_pred_all = svm.predict(X)
    start = time.time()
    y_pred_gt = svm.predict(X_test_gt)
    time_test = time.time()-start
    
    acc = accuracy_score(y_test,y_pred)
    acc_gt = accuracy_score(y_test_gt,y_pred_gt)
    np.savez_compressed(outfile,y_test=y_test,y_pred=y_pred,sh_train=data_train.shape,sh_test=data_test.shape,best_params=[svm.params_['C'],svm.params_['gamma']],acc=[acc],cls_score=[svm.score_],y_pred_all=y_pred_all,y_test_gt=y_test_gt,y_pred_gt=y_pred_gt,acc_gt=[acc_gt],time_train=time_train,time_test=time_test,time_prepare=time_prepare)
    print (outfile,acc,acc_gt,time_train,time_test)

#************************** RUN & SEE  *******************************************

def see_all_tra(name='E(1)',preprocessing='none',stat='acc'):
    """
    presents transductive experiment result

    parameters:
    name: image name
    preprocessing: preprocessing mode
    stat: performance measure ['acc','aa','kappa']

    """
    acc = []
    aa = []
    kappa=[]
    t_train=[]
    t_test=[]
    for i in range(10):
        outfile = 'results/{}_{}_{}.npz'.format(name,i,preprocessing)
        res = np.load(outfile)
        acc.append(res['acc'][0])
        aa.append(calculate_average_accuracy(res['y_test'],res['y_pred']))
        kappa.append(calculate_kappa_coefficient(res['y_test'],res['y_pred']))
        t_train.append(res['time_train']+res['time_prepare'])
        t_test.append(res['time_test']+res['time_prepare'])
    t_train=np.asarray(t_train)
    t_test=np.asarray(t_test)
    res=None
    assert stat in ['acc','aa','kappa']
    res = acc
    if stat=='aa':
        res=aa
    elif stat=='kappa':
        res=kappa
    
    if stat!='kappa':
        print ("{}:{:0.2f}({:0.2f}) , time_train:${:0.1f}\pm{:0.1f}$, time_test:${:0.1f}\pm{:0.1f}$".format(name,np.mean(res)*100
                                                                                                            ,np.std(res)*100
                                                                                                            ,np.mean(t_train)
                                                                                                            ,np.std(t_train)
                                                                                                            ,np.mean(t_test)
                                                                                                            ,np.std(t_test)))
    else:
        print ("{}:{:0.2f}({:0.2f})".format(name,np.mean(res),np.std(res)))
                

def see_all_ind(name_train='F(1)',name_test='E(1)',preprocessing='none',stat='acc'):
    """
    presents inductive experiment result

    parameters:
    name: image name
    preprocessing: preprocessing mode
    stat: performance measure ['acc','aa','kappa']    
    """    
    acc = []
    acc_gt = []
    aa = []
    kappa=[]    
    t_train=[]
    t_test=[]    
    for i in range(10):
        outfile = 'results_i/{}_{}_{}_{}.npz'.format(name_train,name_test,i,preprocessing)
        try:
            res = np.load(outfile)
            acc.append(res['acc'][0])
            acc_gt.append(res['acc_gt'][0])
            aa.append(calculate_average_accuracy(res['y_test_gt'],res['y_pred_gt']))
            kappa.append(calculate_kappa_coefficient(res['y_test_gt'],res['y_pred_gt']))
            t_train.append(res['time_train']+res['time_prepare'])
            t_test.append(res['time_test']+res['time_prepare'])            
        except:
            pass
            #print ("no", outfile)
    t_train=np.asarray(t_train)
    t_test=np.asarray(t_test)
    if len(acc)==0:
        print ("{}->{}: not yet computed".format(name_train,name_test))
        return 1
    res=None
    assert stat in ['acc','aa','kappa']
    res = acc_gt
    if stat=='aa':
        res=aa
    elif stat=='kappa':
        res=kappa

    #print ("{}->{}: {:0.2f}({:0.2f}), all: {:0.2f}({:0.2f})".format(name_train,name_test,np.mean(acc)*100,np.std(acc)*100,np.mean(acc_gt)*100,np.std(acc_gt)*100))    
    if stat=='kappa':
        print ("{}->{}: {:0.2f}({:0.2f})".format(name_train,name_test,np.mean(res),np.std(res)))
    else:
        print ("{}->{}: {:0.2f}({:0.2f}) , time_train:${:0.1f}\pm{:0.1f}$, time_test:${:0.1f}\pm{:0.1f}$".format(name_train
                                                                                                            ,name_test
                                                                                                            ,np.mean(res)*100
                                                                                                            ,np.std(res)*100
                                                                                                            ,np.mean(t_train)
                                                                                                            ,np.std(t_train)
                                                                                                            ,np.mean(t_test)
                                                                                                            ,np.std(t_test)))
    return 0    

      
def fire_inductive(exp_index=0,preprocessing='none'):
    """
    fires inductive experiment for all images

    parameters:
    exp_index: no. experiment (e.g. 0-9)
    preprocessing: preprocessing mode
    """
    for n in DS_I:
        experiment_inductive(name_train=n[0],name_test=n[1],exp_index=exp_index,preprocessing=preprocessing)
        
        
def fire_transductive(exp_index=0,preprocessing='none'):
    """
    fires all transductive experiments

    parameters:
    exp_index: no. experiment (e.g. 0-9)
    preprocessing: preprocessing mode
    """
    for n in DS:
        experiment_transductive(name=n,exp_index=exp_index,preprocessing=preprocessing)    


def results_ind(preprocessing='none',stat='acc'):
    """
    printes results of inductive experiments

    parameters:
    preprocessing: preprocessing mode
    stat: performance measure ['acc','aa','kappa'] 

    """        
    for n in DS_I:    
        see_all_ind(name_train=n[0],name_test=n[1],preprocessing=preprocessing,stat=stat)
        

        
def results_tra(preprocessing='none',stat='acc'):
    """
    printes results of transductive expetimens experiments    

    parameters:
    preprocessing: preprocessing mode
    stat: performance measure ['acc','aa','kappa'] 
    """
    for n in DS:
        see_all_tra(name=n,preprocessing=preprocessing,stat=stat)


if __name__ == '__main__':
    preprocessing='none'
    if True:
        for exp_index in range(10):
            fire_inductive(exp_index,preprocessing=preprocessing)
            fire_transductive(exp_index,preprocessing=preprocessing)
    
    results_ind(preprocessing=preprocessing,stat='acc')
    results_tra(preprocessing=preprocessing,stat='acc')
