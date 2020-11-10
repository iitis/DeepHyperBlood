# Description

Source code for replicating experiments in the paper by
K. Książek, M. Romaszewski, P. Głomb, B. Grabowski, M. Cholewa
`Blood Stains Classification with Hyperspectral Imaging and Deep Neural Networks'

## The dataset associated with the source code:

The dataset is available online: https://zenodo.org/record/3984905

## Usage

* Make sure the paths in data_paths.py are correct
* Run main_train.py to train DNN models, run main_test.py to test them
* Run hyperblood_classification_svm.py for reference SVM experiments 

## License

All files in this repository except: 


* data_dataset.py
* data_loader.py
* main_test.py
* main_train.py
* models.py
* trainer.py
* utils.py                    

are licenced under GNU GENERAL PUBLIC LICENSE Version 3.

However, the above files are based on the code in library: 

https://github.com/nshaud/DeepHyperX

for paper

N. Audebert, B. Le Saux and S. Lefevre, "Deep Learning for Classification of Hyperspectral Data: A Comparative Review,"
in IEEE Geoscience and Remote Sensing Magazine, vol. 7, no. 2, pp. 159-173, June 2019.

The code is used for RESEARCH AND NON COMMERCIAL PURPOSES under the licence:
https://github.com/nshaud/DeepHyperX/blob/master/License