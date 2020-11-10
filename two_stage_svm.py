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

2SCV SVM implementation
"""

import unittest
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris

def get_svm_rbf(X):
    """
    returns classic SVM param grid 
    """
    gamma = 1 / (X.shape[1] * X.var())
    gammas = [gamma*(10**p) for p in np.linspace(-3,3,num=7)]
    param_grid={
                'C': [(10**p) for p in np.linspace(-3,3,num=7)]
                ,'gamma': gammas}
    classifier = SVC(probability=False,kernel='rbf')         
    return classifier,param_grid


class TwoStageSVM():
    def __init__(self,X=None,y=None):
        self.params_=None
        self.score_=None
        self.cls_=None
        if X is not None and y is not None:
            self.fit(X,y)
            
    def fit(self,X,y,n_splits=3):
        """
        fits the model to data with 2SSVM(RBF)
        """
        cls,params=get_svm_rbf(X)
        gs = GridSearchCV(cv=StratifiedKFold(n_splits=n_splits)
                          ,estimator=cls
                          ,param_grid=params
                          ,scoring='accuracy'
                          ,refit=True)
        gs.fit(X,y)
        self.params_=gs.best_params_
        self.score_=gs.best_score_
        self.cls_=gs.best_estimator_
    def predict(self,X):
        """
        predicts labels
        """
        return self.cls_.predict(X)

class Test(unittest.TestCase):
    def test(self):
        
        
        iris = load_iris()
        X=iris.data
        y=np.asarray(iris.target,dtype=np.int32)

        _,params = get_svm_rbf(X)
        print (params)
        
        skf = StratifiedKFold(n_splits=3)
        acc = []
        for train_index, test_index in skf.split(X,y):
            svm = TwoStageSVM(X[train_index],y[train_index])
            #print (svm.params_)
            y_pred = svm.predict(X[test_index])
            acc.append(accuracy_score(y[test_index],y_pred))
        print ("acc: {:0.2f}({:0.2f})".format(np.mean(acc)*100,np.std(acc)*100))
        self.assertGreater(np.mean(acc),0.8)    
            
        
        

if __name__ == '__main__':
    unittest.main()
