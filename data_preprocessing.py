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

File with data preprocessing.
"""

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from sklearn.preprocessing import scale
from scipy import signal
from sklearn import linear_model

# ----------------------------------------------------------------------------

def hsi_median_filter(hsi_image, size):
    return np.dstack([median_filter(hsi_image[:, :, i], size=size) for i in range(hsi_image.shape[2])])

# ----------------------------------------------------------------------------

def spectra_normalisation(X, mode='median'):
    """
    performs spectra normalisation, dividing each spectrum by their median/mean
    
    Arguments:
    X: 2D data array
    mode: per-spectra normalisation[median,mean,max]

    Returns:
    2D array copy with normalised spectra
    """
    X2 = X.copy()
    norms = None
    if mode == 'median':
        norms = np.median(X2, axis=1)
    elif mode == 'mean':
        norms = np.mean(X2, axis=1)
    elif mode == 'max':
        norms = np.max(X2, axis=1)
    else:
        raise NotImplementedError
    for i in range(len(norms)):
        X2[i] = X2[i]/norms[i]
    return X2

# ----------------------------------------------------------------------------

def do_preprocessing(img, preprocessing):
    """
    Function for data preprocessing:
    median filter, first derivative, Savitzky-Golay filter (with or without
    derivatives) or data normalization (scaling)

    Arguments:
    img: image for preprocessing
    preprocessing: type of preprocessing

    Returns:
    img: image after preprocessing
    """
    # Add data preprocessing according to given type

    if preprocessing["type"] == 'median':
        # Median filter
        print("PREPROCESSING: median filter.")
        img = hsi_median_filter(img, preprocessing["median_window"])

    elif preprocessing["type"] == 'derivative':
        # First derivatives
        print("PREPROCESSING: first derivative.")
        try:
            original_img = np.copy(img)
            img = np.zeros((original_img.shape[0],
                            original_img.shape[1],
                            original_img.shape[2] - 1),
                           dtype=np.float32)
            for band in range(0, original_img.shape[2] - 1):
                img[:, :, band] = original_img[:, :, band + 1] - original_img[:, :, band]

        except IndexError as error:
            print("The image has wrong dimensions!")

    elif preprocessing["type"] == 'savitzky':
        # Savitzky-Golay filter
        print("PREPROCESSING: Savitzky-Golay filter.")
        img = savgol_filter(img,
                            window_length=preprocessing["savitzky_window"],
                            polyorder=preprocessing["savitzky_poly"],
                            deriv=preprocessing["savitzky_deriv"],
                            mode=preprocessing["savitzky_mode"])

    elif preprocessing["type"] == 'savitzky_scale':
        # Savitzky-Golay filter + scaling
        print("PREPROCESSING: Savitzky-Golay filter + scaling.")
        img = savgol_filter(img,
                            window_length=preprocessing["savitzky_window"],
                            polyorder=preprocessing["savitzky_poly"],
                            deriv=preprocessing["savitzky_deriv"],
                            mode=preprocessing["savitzky_mode"])
        shape_0, shape_1, shape_2 = img.shape[0], img.shape[1], img.shape[2]
        img = img.reshape((shape_0 * shape_1, shape_2))
        img = scale(img)
        img = img.reshape((shape_0, shape_1, shape_2))

    elif preprocessing["type"] == 'normalization':
        # Normalization
        print("PREPROCESSING: data normalization.")
        shape_0, shape_1, shape_2 = img.shape[0], img.shape[1], img.shape[2]
        img = img.reshape((shape_0 * shape_1, shape_2))
        img = scale(img)
        img = img.reshape((shape_0, shape_1, shape_2))
        # img = (img - np.min(img)) / (np.max(img) - np.min(img))

    elif preprocessing["type"] == "division":
        # Divide spectra by median value
        print("PREPROCESSING: division each spectra by the median value")
        original_shape = img.shape
        img = img.reshape(-1, original_shape[2])
        # division by the median value
        img = spectra_normalisation(img)
        img = img.reshape(original_shape)

    elif preprocessing["type"] == "division_normalization":
        # Divide spectra by median value and normalize data
        print("PREPROCESSING: division each spectra by the median value and data normalization")
        original_shape = img.shape
        img = img.reshape(-1, original_shape[2])
        # division by the median value
        img = spectra_normalisation(img)
        # data scaling
        img = scale(img)
        img = img.reshape(original_shape)

    else:
        print("Data without PREPROCESSING. Nothing will be done.")

    return img
