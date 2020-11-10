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

Main entry point for model training
"""

from utils import get_device
from data_loader import blood_loader
from trainer import train_model
from data_preprocessing import do_preprocessing
from data_paths import PATH_DATA, PATH_SAMPLES

# ----------------------------------------------------------------------------

def get_default_run_options(model, dataset, runs, sampling_mode):
    """Setup general experiment options, irrespective of the model and data.
    Parameters:
        model (str): name of model to use. Available: 
            SVM (linear),  SVM_grid (grid search on  linear, poly and RBF), 
            baseline (fully connected NN), hu (1D CNN),
            hamida (3D CNN + 1D classifier), lee (3D FCN), chen (3D CNN), 
            li (3D CNN), he (3D CNN), luo (3D CNN), sharma (2D CNN), 
            mou (1D RNN) boulch (1D semi-supervised CNN), 
            liu (3D semi-supervised CNN)
        dataset (str): hyperspectral image name.
        runs (int): number of runs.
        sampling_mode ('all' 'fixed'): how to select pixels for train/test.
    Returns:
        options (dict): set of options.
    """
    options = {
        'model': model,
        'runs': runs,
        'sampling_mode': sampling_mode,
        'dataset': dataset,
        'device': get_device(0), # (defaults to -1, which learns on CPU)
        'dataset_path': PATH_DATA,
        'sample_path': PATH_SAMPLES,
        'rdir': 'work/',
        'preprocessing': {'type': 'division'}
        }
    if model == 'hu':
        options['batch_size'], options['epoch'] = 50, 400
    elif model == 'li' or model == 'lee':
        options['batch_size'], options['epoch'] = 100, 200
    else:
        options['batch_size'], options['epoch'] = 100, 100    
    # DeepHyperX default options:
    options['svm_grid_params'] = [
        {'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3], 
             'C': [1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
        {'kernel': ['poly'], 'degree': [3], 'gamma': [1e-1, 1e-2, 1e-3]}]
    options.update({
        'class_balancing': False,
        'flip_augmentation': False,
        'mixture_augmentation': False,
        'multi_class': 1,
        'path': './predictions/',
        'radiation_augmentation': False,
        'test_stride': 1,
        'training_sample': 10,
        'with_exploration': False})
    # DeepHyperX handy, but unused options
    options.update({
        'checkpoint': None, # option to load state dict instead of train from scratch
        'train_gt': None, # train GT filename, not used
        'test_gt': None, # test GT filename, not used
        })
    return options

# ----------------------------------------------------------------------------

def load_and_update(options):
    """Load dataset, and update options."""
    img, gt, _, ignored_labels, label_values = \
        blood_loader(options['dataset_path'], options['dataset'])
    img = do_preprocessing(img, options['preprocessing'])
    options.update({'n_classes': len(label_values),
                    'n_bands': img.shape[-1],
                    'ignored_labels': ignored_labels,
                    'label_values': label_values})
    return img, gt    

# ----------------------------------------------------------------------------

def run_train(datasets=['E_1', 'E_7', 'E_21', 'F_1', 'F_1a', 
                        'F_2', 'F_2k', 'F_7', 'F_21'],
              models=['nn', 'hu', 'lee', 'li', 'hamida', 'mou'],
              runs=10):
    """Run a sequence of training for DeepHyperBlood experiments.
    Parameters:
        datasets (list of strings): images to run experiments on.
        models (list of strings): models to be evaluated.
        runs (int): number of runs.
    Returns:
        (Nothing, the trained models are saved as a file)
    """
    for dataset in datasets:
        for model in models:
            options = get_default_run_options(model, dataset, runs,
                                              sampling_mode='fixed')
            img, gt = load_and_update(options)
            train_model(img, gt, options)

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run_train(datasets=['E_1'], models=['hu'], runs=1)
