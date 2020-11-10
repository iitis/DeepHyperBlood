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
"""
# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division

# Torch
import torch
import torch.utils.data as data
from torchsummary import summary
import numpy as np
import sklearn.svm
import sklearn.model_selection

from utils import open_file, metrics, sample_gt, build_dataset, \
        compute_imf_weights
from data_dataset import HyperX
from models import get_model, train, test, save_model
from data_split import select_subset, check_split_correctness, get_fixed_sets

# ----------------------------------------------------------------------------

def train_model(img, gt, hyperparams):
    """
    Function for model training.
    1) Data sampling into a training, a validation and a test set.
    2) Training a chosen model.
    3) Model evaluation.

    Arguments:
    img - dataset (hyperspectral image)
    gt - ground truth (labels)
    hyperparams - parameters of training
    SVM_GRID_PARAMS - parameters for SVM (if used)
    FOLDER - a path for datasets
    DATASET - name of the used dataset 
    set_parameters: option for loading a specific training and test set
    preprocessing_parameters: parameters of preprocessing
    """
    print("img.shape: {}".format(img.shape))
    print("gt.shape: {}".format(gt.shape))

    # all images should have 113 bands
    assert(img.shape[2] == 113)

    viz = None
    results = []
    # run the experiment several times
    for run in range(hyperparams['runs']):
        #############################################################################
        # Create a training and a test set
        if hyperparams['train_gt'] is not None and hyperparams['test_gt'] is not None:
            train_gt = open_file(hyperparams['train_gt'])
            test_gt = open_file(hyperparams['test_gt'])
        elif hyperparams['train_gt'] is not None:
            train_gt = open_file(hyperparams['train_gt'])
            test_gt = np.copy(gt)
            w, h = test_gt.shape
            test_gt[(train_gt > 0)[:w, :h]] = 0
        elif hyperparams['test_gt'] is not None:
            test_gt = open_file(hyperparams['test_gt'])
        else:
            # Choose type of data sampling
            if hyperparams['sampling_mode'] == 'uniform':
                train_gt, test_gt = select_subset(gt, hyperparams['training_sample'])
                check_split_correctness(gt, train_gt, test_gt, hyperparams['n_classes'])
            elif hyperparams['sampling_mode'] == 'fixed':
                # load fixed sets from a given path
                train_gt, test_gt = get_fixed_sets(run, hyperparams['sample_path'], hyperparams['dataset'])
                check_split_correctness(gt, train_gt, test_gt, hyperparams['n_classes'], 'fixed')
            else:
                train_gt, test_gt = sample_gt(gt,
                                              hyperparams['training_sample'],
                                              mode=hyperparams['sampling_mode'])
            
        print("{} samples selected (over {})".format(np.count_nonzero(train_gt),
                                                     np.count_nonzero(gt)))
        print("Running an experiment with the {} model".format(hyperparams['model']),
              "run {}/{}".format(run + 1, hyperparams['runs']))
        #######################################################################
        # Train a model

        if hyperparams['model'] == 'SVM_grid':
            print("Running a grid search SVM")
            # Grid search SVM (linear and RBF)
            X_train, y_train = build_dataset(img, train_gt,
                                             ignored_labels=hyperparams['ignored_labels'])
            class_weight = 'balanced' if hyperparams['class_balancing'] else None
            clf = sklearn.svm.SVC(class_weight=class_weight)
            clf = sklearn.model_selection.GridSearchCV(clf,
                                                       hyperparams['svm_grid_params'],
                                                       verbose=5,
                                                       n_jobs=4)
            clf.fit(X_train, y_train)
            print("SVM best parameters : {}".format(clf.best_params_))
            prediction = clf.predict(img.reshape(-1, hyperparams['n_bands']))
            save_model(clf,
                       hyperparams['model'],
                       hyperparams['dataset'],
                       hyperparams['rdir'])
            prediction = prediction.reshape(img.shape[:2])
        elif hyperparams['model'] == 'SVM':
            X_train, y_train = build_dataset(img, train_gt,
                                             ignored_labels=hyperparams['ignored_labels'])
            class_weight = 'balanced' if hyperparams['class_balancing'] else None
            clf = sklearn.svm.SVC(class_weight=class_weight)
            clf.fit(X_train, y_train)
            save_model(clf,
                       hyperparams['model'],
                       hyperparams['dataset'],
                       hyperparams['rdir'])
            prediction = clf.predict(img.reshape(-1, hyperparams['n_bands']))
            prediction = prediction.reshape(img.shape[:2])
        elif hyperparams['model'] == 'SGD':
            X_train, y_train = build_dataset(img, train_gt,
                                             ignored_labels=hyperparams['ignored_labels'])
            X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
            scaler = sklearn.preprocessing.StandardScaler()
            X_train = scaler.fit_transform(X_train)
            class_weight = 'balanced' if hyperparams['class_balancing'] else None
            clf = sklearn.linear_model.SGDClassifier(class_weight=class_weight,
                                                     learning_rate='optimal',
                                                     tol=1e-3,
                                                     average=10)
            clf.fit(X_train, y_train)
            save_model(clf,
                       hyperparams['model'],
                       hyperparams['dataset'],
                       hyperparams['rdir'])
            prediction = clf.predict(scaler.transform(img.reshape(-1,
                                                      hyperparams['n_bands'])))
            prediction = prediction.reshape(img.shape[:2])
        elif hyperparams['model'] == 'nearest':
            X_train, y_train = build_dataset(img,
                                             train_gt,
                                             ignored_labels=hyperparams['ignored_labels'])
            X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
            class_weight = 'balanced' if hyperparams['class_balancing'] else None
            clf = sklearn.neighbors.KNeighborsClassifier(weights='distance')
            clf = sklearn.model_selection.GridSearchCV(clf,
                                                       {'n_neighbors': [1, 3, 5, 10, 20]},
                                                       verbose=5,
                                                       n_jobs=4)
            clf.fit(X_train, y_train)
            clf.fit(X_train, y_train)
            save_model(clf,
                       hyperparams['model'],
                       hyperparams['dataset'],
                       hyperparams['rdir'])
            prediction = clf.predict(img.reshape(-1, hyperparams['n_bands']))
            prediction = prediction.reshape(img.shape[:2])
        else:
            # Neural network
            model, optimizer, loss, hyperparams = get_model(hyperparams['model'], **hyperparams)
            if hyperparams['class_balancing']:
                weights = compute_imf_weights(train_gt,
                                              hyperparams['n_classes'],
                                              hyperparams['ignored_labels'])
                hyperparams['weights'] = torch.from_numpy(weights)
            # Split train set in train/val
            if hyperparams['sampling_mode'] in {'uniform', 'fixed'}:
                train_gt, val_gt = select_subset(train_gt, 0.95)
            else:
                train_gt, val_gt = sample_gt(train_gt, 0.95, mode='random')
            # Generate the dataset
            train_dataset = HyperX(img, train_gt, **hyperparams)
            train_loader = data.DataLoader(train_dataset,
                                           batch_size=hyperparams['batch_size'],
                                           shuffle=True)
            val_dataset = HyperX(img, val_gt, **hyperparams)
            val_loader = data.DataLoader(val_dataset,
                                         batch_size=hyperparams['batch_size'])

            print(hyperparams)
            print("Network :")
            with torch.no_grad():
                for input, _ in train_loader:
                    break
                summary(model.to(hyperparams['device']), input.size()[1:])
                # We would like to use device=hyperparams['device'] altough we have
                # to wait for torchsummary to be fixed first.

            if hyperparams['checkpoint'] is not None:
                model.load_state_dict(torch.load(hyperparams['checkpoint']))

            try:
                train(model,
                      optimizer,
                      loss,
                      train_loader,
                      hyperparams['epoch'],
                      scheduler=hyperparams['scheduler'],
                      device=hyperparams['device'],
                      supervision=hyperparams['supervision'],
                      val_loader=val_loader,
                      display=viz,
                      rdir=hyperparams['rdir'],
                      model_name=hyperparams['model'],
                      preprocessing=hyperparams['preprocessing']['type'],
                      run=run)
            except KeyboardInterrupt:
                # Allow the user to stop the training
                pass

            probabilities = test(model, img, hyperparams)
            prediction = np.argmax(probabilities, axis=-1)

        #######################################################################
        # Evaluate the model
        # If test set is not empty
        if(np.unique(test_gt).shape[0] > 1):
            run_results = metrics(prediction,
                                  test_gt,
                                  ignored_labels=hyperparams['ignored_labels'],
                                  n_classes=hyperparams['n_classes'])

        mask = np.zeros(gt.shape, dtype='bool')
        for l in hyperparams['ignored_labels']:
            mask[gt == l] = True
        prediction[mask] = 0

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
