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

File with data splitting for stratifying the image during training.
"""
import numpy as np
import matplotlib.pyplot as plt
from utils import open_file

# ----------------------------------------------------------------------------

def get_fixed_sets(run, folder_name, dataset_name, mode='both'):
    """
    Load fixed training and test set from the file.

    Arguments:
    run - number of a current run
    folder_name - folder with files
    dataset_name - name of a loaded dataset
    mode - 'both': train and test set,
           'train': only training set
           'test': only test set
    """
    if mode in ('both', 'train'):
        train_gt = open_file(folder_name + dataset_name + '/train_gt_' + str(run) + '.npz')
        if mode == 'train':
            return train_gt

    if mode in ('both', 'test'):
        test_gt = open_file(folder_name +  dataset_name + '/test_gt_' + str(run) + '.npz')
        if mode == 'test':
            return test_gt
        else:
            return train_gt, test_gt

    else:
        print('Wrong name of the mode parameter!')
        pass

# ----------------------------------------------------------------------------

def select_subset(gt, size=500, min_n_class=5):
    """
    stratified selection of a random subset from image

    parameters:
    gt: labels
    size: if size>1: no. examples/class, if size<1: ratio of examples/class
    min_n_class: minimum number of examples/class (if exists)

    returns:
    training set, test set
    """
    all_gt_classes = np.unique(gt)
    trash_classes = np.array([0, 255])
    # all classes which are in ground truth
    classes = np.setdiff1d(all_gt_classes, trash_classes)

    indices_train, indices_test = [], []
    train_set = np.zeros_like(gt)
    test_set = np.zeros_like(gt)
    # select samples for each class
    for current_class in classes:
        indices = np.nonzero(gt == current_class)
        X = list(zip(*indices))
        np.random.shuffle(X)

        # number of elements of current_class to seek
        to_seek = int(size) if size > 1 else int(len(indices[0]) * size)
        if to_seek < min_n_class:
            to_seek = np.min([min_n_class,
                              np.count_nonzero(gt == current_class)])
        train_sample = X[0:to_seek]
        indices_train = indices_train + train_sample
        test_sample = X[to_seek:]
        indices_test = indices_test + test_sample

    # prepare tables with training and test set
    for element in indices_train:
        train_set[element] = gt[element]

    for element in indices_test:
        test_set[element] = gt[element]

    return train_set, test_set

# ----------------------------------------------------------------------------

def check_split_correctness(validation_gt, train_gt, test_gt, no_of_classes=6, mode='general'):
    """
    Function for checking the correctness of the set split

    mode - 'general': division into the training and the test set inside DeepHyperX
           'fixed': loading prepared files
    """
    if mode != 'fixed':
        # first level: does sum of train and test set equal to whole gt?
        check_gt = np.array(train_gt + test_gt)
        diff = validation_gt - check_gt
        assert((diff == 0).all())
        assert((check_gt == validation_gt).all())

    classes = np.arange(1, no_of_classes + 1, 1)
    for no in classes:
        test_class = np.count_nonzero(train_gt == no)
        check_class = np.count_nonzero(test_gt == no)
        print('In train set are {} samples of {} class.'.format(test_class, no))
        print('In test set are {} samples of {} class.'.format(check_class, no))
    no_of_border = np.count_nonzero(train_gt == 255)
    no_of_classes_train = np.count_nonzero(train_gt) - no_of_border
    print('Length of training set: {}'.format(no_of_classes_train))
    print('Length of test set: {}'.format(np.count_nonzero(test_gt)))
    print('Ratio of non-zero elements in training set to the non-zero elements'
          ' in the whole set: {}'.format(
            no_of_classes_train / (no_of_classes_train + np.count_nonzero(test_gt))
            ))

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    pass