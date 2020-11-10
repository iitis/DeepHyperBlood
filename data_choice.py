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

File for creation of training and test sets.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.utils import shuffle
from data_loader import blood_loader
from data_paths import PATH_DATA

# ----------------------------------------------------------------------------

def patch_substance_checking(substance_occurences,
                             substance,
                             indices,
                             patch_size,
                             mode='train',
                             train_indices=None):
    '''
    Check if a pixel could be attached into the training or the test set.

    Arguments:
        substance_occurences - places where the given substance is present
        substance - label for the substance
        indices - position of a given pixel
        patch_size - size of a patch for network
        mode - 'train' or 'test'
        train_indices - indices of the training set (only for 'test' mode)

    Returns:
        True/False (is the pixel admissible or not)
    '''
    # check if the patch is inside the substance
    shapes = substance_occurences.shape
    # patch size should be an odd value
    if patch_size % 2 != 0:
        limit = patch_size // 2
    else:
        print('Patch size should be an odd value!')
        pass
    # if in all pixels in the window the substance is present, return True
    # in the other case, the function returns False
    if mode == 'train':
        for i in range(-limit, limit + 1):
            for j in range(-limit, limit + 1):
                # if a pixel is in the possible range
                if((indices[0] + i < shapes[0]) and (indices[1] + j < shapes[1])):
                    continue
                else:
                    return False
        return True

    # check if it is possible to join the pixel into the test set
    elif mode == 'test':
        for i in range(-limit, limit + 1):
            for j in range(-limit, limit + 1):
                if((indices[0] + i < shapes[0]) and (indices[1] + j < shapes[1])):
                    # check if any pixel from the patch window is in the
                    # training set
                    if(train_indices[indices[0] + i][indices[1] + j]) == substance:
                        return False
                else:
                    return False
        return True

    else:
        return False

# ----------------------------------------------------------------------------

def get_train_indices(image,
                      classes,
                      no_of_train_samples,
                      patch_size):
    '''
    Random choice of a training set

    Arguments:
        image - the original ground truth
        classes - all classes to find
        no_of_train_samples - number of training samples for each class
        patch_size - size of a patch for network

    Returns:
        train_indices - indices of the training set
    '''
    # an array for indices of training samples
    train_indices = np.zeros_like(image)
    for substance in classes:
        # find all occurences of a given substance
        substance_occurences = np.copy(image)
        for x in np.nditer(substance_occurences, op_flags=['readwrite']):
            x[...] = 1 if x == substance else 0
        # find indices of a given substance
        substance_indices = np.argwhere(substance_occurences == 1)
        order_indices = shuffle(substance_indices)
        # get indices of a given substance, if it is possible
        counter = 0
        for indices in order_indices:
            # check if the pixel located in this position can be attached
            if(patch_substance_checking(substance_occurences,
                                        1,
                                        indices,
                                        patch_size,
                                        mode='train')):
                train_indices[tuple(indices)] = 1
                counter += 1
            # when a given number of samples is prepared, stop the algorithm
            if(counter == no_of_train_samples):
                break
        if(counter < no_of_train_samples):
            print(f'The choice of {no_of_train_samples} is impossible! \
                    Only {counter} samples were chosen!')
    return train_indices

# ----------------------------------------------------------------------------

def get_test_indices(image,
                     train_indices,
                     patch_size):
    '''
    Choice of a test set where intersection with the training samples is empty

    Arguments:
        image - the original ground truth
        train_indices - indices of the training set
        patch_size - size of a patch for network

    Returns:
        test_indices - indices of the test set
    '''
    # get indices of a test set
    # change indices of all substances to 1
    all_substances_occurences = np.where(image != 0, 1, 0)
    # select pixels which are free
    free_substances_occurences = all_substances_occurences - train_indices
    plt.imshow(free_substances_occurences)
    plt.savefig('free_indices.png', dpi=200)
    free_substances_indices = np.argwhere(free_substances_occurences == 1)

    test_indices = np.zeros_like(image)
    for indices in free_substances_indices:
        # check if it is possible to join pixel located in this position
        # into the test set
        if(patch_substance_checking(free_substances_occurences,
                                    1,
                                    indices,
                                    patch_size,
                                    mode='test',
                                    train_indices=train_indices)):
            test_indices[tuple(indices)] = 1
    return test_indices

# ----------------------------------------------------------------------------

def specify_classes_and_samples(image,
                                parameter):
    '''
    Count number of classes present in the image and designate the number
    of samples per each class.

    Arguments:
        image - the original ground truth
        parameter - percentage of the least numerous class

    Return:
        classes - indices of classes (without background which is equal to 0!)
        no_of_train_samples - number of training samples per each class
    '''
    # count samples of a given class and find a minimum of them
    # from 1st index because 0 is background
    classes = np.unique(image)[1:]
    minimum = image.shape[0] * image.shape[1]

    for substance in classes:
        count_substance = np.count_nonzero(image == substance)
        print(f'no of samples of {substance}: {count_substance}')
        # if this class is the least numerous one
        if count_substance < minimum:
            minimum = count_substance
    print(f'The least numerous class has {minimum} samples.')
    no_of_train_samples = int(parameter * minimum)
    print(f'Number of samples from each class in training set will be {no_of_train_samples}')
    return classes, no_of_train_samples

# ----------------------------------------------------------------------------

def prepare_final_sets(image,
                       indices):
    '''
    Prepare a final ground truth of given indices.
    In each place for training (test) set a label of the substance will
    be present.

    Arguments:
        image - the original ground truth
        indices - image: in training (test) set place is 1, 0 in the other case

    Returns:
        final_ground_truth - 2D ground truth image for the training (test) set
    '''
    original_shape = image.shape
    ground_truth = np.copy(image).reshape(-1)
    indices_ground_truth = np.copy(indices).reshape(-1)
    final_ground_truth = np.zeros(original_shape).reshape(-1)

    for i in range(0, len(ground_truth)):
        if(indices_ground_truth[i] == 1):
            final_ground_truth[i] = ground_truth[i]

    final_ground_truth = final_ground_truth.reshape(original_shape)
    return final_ground_truth

# ----------------------------------------------------------------------------

def test_intersection_train_test(image,
                                 train_indices,
                                 test_indices,
                                 parameter,
                                 path,
                                 run):
    '''
    Unit test: check if intersection train and test set is an empty set

    Arguments:
        image - the original ground truth
        train_indices - indices of a training set
        test_indices - indices of a test set
        parameter - percentage of the least numerous class
        path -
        run -
    '''
    union_indices = train_indices + test_indices
    # maximum of union of two sets should be equal to one
    assert(np.max(union_indices) == 1)
    print('Test of intersection passed.')

    # 0 - background
    joint_set = np.copy(train_indices)  # 1 - training set
    test_for_joint_set = np.where(test_indices == 1, 2, 0)  # 2 - training set
    image = np.where(image != 0, 1, 0)
    unused = image - train_indices - test_indices
    unused_for_joint_set = np.where(unused == 1, 3, 0)  # 3 - unused
    joint_set = joint_set + test_for_joint_set + unused_for_joint_set

    color_dict = {0: [0.1, 0.0, 0.6, 1.0],
                  1: [1.0, 1.0, 0.0, 1.0],
                  2: [0.5, 0.9, 1.0, 1.0],
                  3: [1.0, 0.0, 0.0, 1.0]}
    list_of_colors = [[i/3, color_dict[i]] for i in range(0, 4)]
    cmap = LinearSegmentedColormap.from_list('mycmap', list_of_colors)
    labels = {0: 'background',
              1: 'training set',
              2: 'test set',
              3: 'unused'}
    image_show = plt.imshow(joint_set, cmap=cmap, vmin=0, vmax=3)
    patches = [mpatches.Patch(color=color_dict[i], label=labels[i]) for i in color_dict]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 0.75), loc=2, borderaxespad=0.)
    plt.savefig(f'{path}/summup_{parameter}_{run}.png', bbox_inches='tight', dpi=200)
    plt.clf()

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    images_full_names = ['E_7']
    images = ['scene_comparison_day_7']

    for image_full_name, name in zip(images_full_names, images):
        path = f"./Fixed_sets_preparing/{name}"
        os.makedirs(path, exist_ok=True)
        _, image, _, _, _ = blood_loader(PATH_DATA, image_full_name)
        patch_size = 5
        # 5% of the least numerous class size is the number of samples
        # from following classes
        parameter = 0.05
        N = 10
        # get information about classes and the number of training samples
        classes, no_of_train_samples = specify_classes_and_samples(image,
                                                                   parameter)
        # N sets will be prepared
        for run in range(N):
            # Choose randomly the training set
            train_indices = get_train_indices(image,
                                              classes,
                                              no_of_train_samples,
                                              patch_size)
            plt.imshow(train_indices)
            plt.savefig(f'{path}/train_indices_{parameter}_{run}.png', dpi=200)
            # Choose randomly the test set
            test_indices = get_test_indices(image,
                                            train_indices,
                                            patch_size)
            plt.imshow(test_indices)
            plt.savefig(f'{path}/test_indices_{parameter}_{run}.png', dpi=200)
            # Unittest
            test_intersection_train_test(image,
                                         train_indices,
                                         test_indices,
                                         parameter,
                                         path,
                                         run)

            # Print the number of samples
            train_samples = np.count_nonzero(train_indices == 1)
            test_samples = np.count_nonzero(test_indices == 1)
            print(f'Length of the training set: {train_samples}, test set: {test_samples}')
            print('Ratio of non-zero elements in training set to the non-zero '
                   'elements in the whole set: {}'.format(
                   train_samples / (train_samples + test_samples)))
            train_gt = prepare_final_sets(image, train_indices)
            test_gt = prepare_final_sets(image, test_indices)
            np.savez(f'{path}/train_gt_{run}.npz', gt=train_gt)
            np.savez(f'{path}/test_gt_{run}.npz', gt=test_gt)
