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

Data loader
"""

import unittest
import numpy as np
from data_library import get_data, get_anno
from data_paths import PATH_DATA

# ----------------------------------------------------------------------------

def blood_loader(path, name):
    """
    returns data and annotation, removing classes that are not present in all images
    
    Arguments:
    path: data path
    name: image name

    Returns:
    img: data cube
    gt: annotation with correct classes
    rgb_bands: three bands used for rgb visualisation
    ignored_labels: labels that should be ignored [0 for background] 
    label_values: class labels
    """
    img = np.asarray(get_data(name, path=path)[0], dtype='float32')
    gt = get_anno(name, path=path).astype('uint8')
    # remove beetroot juice (is only on frames images so in our
    # classification experiments we removed it from pictures)
    gt = np.where(gt == 4, 0, gt)
    # renumbering after removing beetroot juice
    for element in [5, 6, 7]:
        gt = np.where(gt == element, element - 1, gt)
    label_values = ["unclassified",
                    "blood",
                    "ketchup",
                    "artificial blood",
                    "poster paint",
                    "tomato concentrate",
                    "acrylic paint"]
    rgb_bands, ignored_labels = (47, 31, 15), [0]
    return img, gt, rgb_bands, ignored_labels, label_values

# ----------------------------------------------------------------------------

class LoadTest(unittest.TestCase):
    def test_load(self):
        img, gt, _,_,_ = blood_loader(PATH_DATA,'F(1)')
        self.assertSequenceEqual(img.shape,(519, 696, 113))
        self.assertSequenceEqual(np.unique(gt).tolist(),[0,1,2,3,4,5,6])


if __name__ == "__main__":
    unittest.main()
