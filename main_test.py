# -*- coding: utf-8 -*-
"""
************************************************************************
Copyright 2020 Institute of Theoretical and Applied Informatics, 
Polish Academy of Sciences (ITAI PAS) https://www.iitis.pl
author: K. Książek, P.Głomb, M. Romaszewski

The code in this file is based on the code from library: https://github.com/nshaud/DeepHyperX
for paper
N. Audebert, B. Le Saux and S. Lefevre, "Deep Learning for Classification of Hyperspectral Data: A Comparative Review,"
in IEEE Geoscience and Remote Sensing Magazine, vol. 7, no. 2, pp. 159-173, June 2019.

The code is used for RESEARCH AND NON COMMERCIAL PURPOSES under the licence:
https://github.com/nshaud/DeepHyperX/blob/master/License
Therefore, the original authors license is used for the code in this file.
************************************************************************

Code for experiments in the paper by  
K. Książek, M. Romaszewski, P. Głomb, B. Grabowski, M. Cholewa
`Blood Stains Classification with Hyperspectral
Imaging and Deep Neural Networks'

Main entry point for testing
"""
import joblib
import os
import torch
import numpy as np

from models import get_model, test
from plots import create_plots
from utils import metrics, show_results
from data_split import get_fixed_sets
from main_train import get_default_run_options, load_and_update

# ----------------------------------------------------------------------------

def get_checkpoint_filename(hyperparams):
    fname = '{rdir}/checkpoint_{train_dataset}'.format(**hyperparams)
    fname += '_{}_'.format(hyperparams['preprocessing']['type'])
    fname += '{model}_{run}_epoch_{epoch}_batch_{batch_size}.pth'.format(
        **hyperparams)
    return fname

# ----------------------------------------------------------------------------

def model_prediction(img, fname, hyperparams):
    if hyperparams['model'] in ['SVM', 'SVM_grid', 'SGD', 'nearest']:
        model = joblib.load(get_checkpoint_filename(hyperparams))
        X = img.reshape((-1, img.shape[-1]))
        prediction = model.predict(X)
        prediction = prediction.reshape(img.shape[:2])
    else:
        model, _, _, hyperparams = get_model(hyperparams['model'], 
                                             **hyperparams)
        model.load_state_dict(torch.load(get_checkpoint_filename(hyperparams)))
        probabilities = test(model, img, hyperparams)
        np.save(fname + "_probabilities", probabilities)
        prediction = np.argmax(probabilities, axis=-1)
    np.save(fname + "_prediction", prediction)
    return prediction, hyperparams

# ----------------------------------------------------------------------------

def single_run_test(hyperparams):
    img, gt = load_and_update(hyperparams)
    ofname = get_checkpoint_filename(hyperparams)[:-4] 
    ofname += "_" + hyperparams['dataset']
    prediction, hyperparams = model_prediction(img, ofname, hyperparams)

    if hyperparams['sampling_mode'] == 'fixed':
        gt = get_fixed_sets(hyperparams['run'], hyperparams['sample_path'], 
                            hyperparams['dataset'], mode='test')    
    run_results = metrics(prediction, gt, hyperparams['ignored_labels'], 
                          hyperparams['n_classes'])
    path = '{rdir}/prediction_training_{train_dataset}_test_{dataset}_epoch_{epoch}_batch_{batch_size}'.format(**hyperparams)
    os.makedirs(path, exist_ok=True)
    show_results(
        run_results,
        None,
        hyperparams['model'],
        hyperparams['dataset'],
        path,
        hyperparams['preprocessing']["type"],
        label_values=hyperparams['label_values'],
        training_image=hyperparams['train_dataset'],
        agregated=False)
    plot_names = {
        'path': path,
        'checkpoint': get_checkpoint_filename(hyperparams),
        'dataset': hyperparams['dataset'],
        'ignored': hyperparams['ignored_labels']
    }
    create_plots(hyperparams['multi_class'], prediction, gt, plot_names)

# ----------------------------------------------------------------------------

def run_test(dataset_pairs,
             sampling_mode, # 'fixed' or 'all'
             models=['nn', 'hu', 'lee', 'li', 'hamida', 'mou'],
             runs=10):
    for train_dataset, test_dataset in dataset_pairs:
        for model in models:
            for run in range(runs):
                options = get_default_run_options(model, test_dataset, runs,
                                                  sampling_mode)
                options['run'], options['train_dataset'] = run, train_dataset
                single_run_test(options)

# ----------------------------------------------------------------------------

def run_test_all():
    ds = ['E_1', 'E_7', 'E_21', 'F_1', 'F_1a', 'F_2', 'F_2k', 'F_7', 'F_21']
    ds_t = [(d, d) for d in ds]
    run_test(ds_t, sampling_mode='fixed')
    ds_i = [('F_1', 'E_1'), ('F_1a', 'E_1'), ('F_2', 'E_7'), ('F_2k', 'E_7'),
            ('F_7', 'E_7'), ('F_21', 'E_21'), ('F_2', 'F_2k'), ('F_2k', 'F_2')]
    run_test(ds_i, sampling_mode='all')

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    #run_test_all()
    run_test(dataset_pairs=[('E_1', 'E_1')], sampling_mode='all', 
             models=['nn'], runs=1)
