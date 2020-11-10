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

HyperBlood API
Basic loader for dataset files for
HSI blood classification dataset by M. Romaszewski, P.Glomb, M. Cholewa, A. Sochan 
Institute of Theoretical and Applied Informatics, Polish Academy of Sciences (ITAI PAS) https://www.iitis.pl
Dataset DOI: 10.5281/zenodo.3984905

Warning:
    * By default, data is cleared by removing noisy bands and broken line in the image. 
    * Note that the 'F(2k)' image was captured with different camera. Its bands were interpolated 
    to match remaining images. However, due to spectral range differences between cameras, it has 
    less bands. After cleaning (default) all images have the same matching 113 bands. 

NOISY_BANDS_INDICES = np.array([0,1,2,3,4,48,49,50,121,122,123,124,125,126,127])
"""
import unittest
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
from data_paths import PATH_DATA

IMAGES = ['A(1)','B(1)','C(1)','D(1)','E(1)','E(7)','E(21)','F(1)','F(1a)','F(1s)','F(2)','F(2k)','F(7)','F(21)'] 


#------------------------ DATA LOADING ------------------------------------

def get_data(name,remove_bands=True,clean=True, path=PATH_DATA):
    """
    Returns HSI data from a datacube
    
    Parameters:
    ---------------------
    name: name
    remove_bands: if True, noisy bands are removed (leaving 113 bands)
    clean: if True, remove damaged line
    
    Returns:
    -----------------------
    data, wavelenghts as numpy arrays (float32)
    """
    name = convert_name(name)
    filename = "{}data/{}".format(path,name)
    hsimage = envi.open('{}.hdr'.format(filename),'{}.float'.format(filename))
    wavs = np.asarray(hsimage.bands.centers) 
    data = np.asarray(hsimage[:,:,:],dtype=np.float32)
    
    #removal of damaged sensor line
    if clean and name!='F_2k':
        data = np.delete(data,445,0)
    
    
    if not remove_bands:
        return data,wavs
    return data[:,:,get_good_indices(name)],wavs[get_good_indices(name)] 

def get_anno(name,remove_uncertain_blood=True,clean=True, path=PATH_DATA):
    """
    Returns annotation (GT) for data files as 2D int numpy array
    Classes:
    0 - background
    1 - blood
    2 - ketchup
    3 - artificial blood
    4 - beetroot juice
    5 - poster paint
    6 - tomato concentrate
    7 - acrtylic paint
    8 - uncertain blood
    
    Parameters:
    ---------------------
    name: name
    clean: if True, remove damaged line
    remove_uncertain_blood: if True, removes class 8 
    
    Returns:
    -----------------------
    annotation as numpy 2D array 
    """
    name = convert_name(name)
    filename = "{}anno/{}".format(path,name)
    anno = np.load(filename+'.npz')['gt']
    #removal of damaged sensor line
    if clean and name!='F_2k':
        anno = np.delete(anno,445,0)
    #remove uncertain blood + technical classes
    if remove_uncertain_blood:
        anno[anno>7]=0 
    else:
        anno[anno>8]=0
           
    return anno    


#------------------------ UTILITY ------------------------------------


def get_good_indices(name=None):
    """
    Returns indices of bands which are not noisy

    Parameters:
    ---------------------
    name: name
    Returns:
    -----------------------
    numpy array of good indices         
    """
    name = convert_name(name)
    if name!='F_2k':
        indices = np.arange(128)
        indices = indices[5:-7]
    else:
        indices = np.arange(116)    
    indices=np.delete(indices,[43,44,45])
    return indices

def convert_name(name):
    """
    Ensures that the name is in the filename format
    Parameters:
    ---------------------
    name: name
    
    Returns:
    -----------------------
    cleaned name
    """
    name = name.replace('(','_')
    name = name.replace(')','')
    return name



def get_rgb(data,wavelengths,gamma=0.7,vnir_bands=[600, 550, 450]):
    """
   Treturns an (over)simplified RGB visualization of HSI data
    
    Parameters:
    ---------------------
    data: data cube as nparray
    annotation: wavelengths - band wavelenghts
    gamma: gamma correction value
    vnir_bands: bands used for RGB
    
    Returns:
    -----------------------
    rgb image as numpy array     
    """
    assert data.shape[2]==len(wavelengths)
    max_data = np.max(data)
    rgb_i = [np.argmin(np.abs(wavelengths - b)) for b in vnir_bands]
    ret = data[:,:,rgb_i].copy()/max_data

    if gamma!=1.0:
        for i in range(3):
            ret[:,:,i]=np.power(ret[:,:,i],gamma)
    
    return ret 

class LoadTest(unittest.TestCase):
    def test_load(self):
        """
        test image loading
        """
        for name in IMAGES:
            data,wavelengths = get_data(name,remove_bands=True)
            anno = get_anno(name)
            self.assertEqual(data.shape[2],113)
            self.assertEqual(data.shape[2],wavelengths.shape[0])
            rgb = get_rgb(data,wavelengths)
            plt.subplot(1,2,1)
            plt.imshow(rgb,interpolation='nearest')
            plt.subplot(1,2,2)
            plt.imshow(anno,interpolation='nearest')
            plt.show()
            plt.close()
            

    def dis_test_indices(self):    
        '''
        Ensure F_2k is loaded correctly 
        '''
        _,wavs = get_data('F_2k',remove_bands=False)
        assert 619.7518 in wavs 
        _,wavs = get_data('F_2k',remove_bands=True)
        assert 619.7518 not in wavs
        _,wavs2 = get_data('F_1',remove_bands=True)
        assert np.sum(wavs-wavs2)==0
        
        data,wavelengths = get_data('F_1',remove_bands=False)
        self.assertEqual(data.shape[2],128)
        self.assertEqual(data.shape[2],wavelengths.shape[0])
        data,wavelengths = get_data('F_2k',remove_bands=False)
        self.assertEqual(data.shape[2],116)
        self.assertEqual(data.shape[2],wavelengths.shape[0])
        anno = get_anno('F_1')
        
        
if __name__ == '__main__':
    unittest.main()
