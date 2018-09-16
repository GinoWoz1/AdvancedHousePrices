# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 07:22:30 2018

@author: jjonus
"""

from sklearn.metrics import make_scorer
from tpot import TPOTRegressor
import warnings
import pandas as pd
import math
import os
import numpy as np

mingw_path = 'C:\\Users\\jstnjc\\Anaconda3\\Library\\mingw-w64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

warnings.filterwarnings('ignore')

url = 'https://github.com/GinoWoz1/AdvancedHousePrices/raw/master/'

X_train = pd.read_csv(url + 'x_train_155_noEng.csv')
X_train.drop(['Unnamed: 0'],axis=1,inplace=True)

y_train = pd.read_csv(url + 'y_train.csv', header=None)
y_train.rename(columns={0:'SalePrice'},inplace=True)
y_train['SalePrice'] = y_train['SalePrice'].astype(float)

def rmsle_loss(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    try:
        terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y_true[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    except:
        return float('inf')
    if not (y_true >= 0).all() and not (y_pred >= 0).all():
            return float('inf')
    return (sum(terms_to_sum) * (1.0/len(y_true))) ** 0.5

rmsle_loss = make_scorer(rmsle_loss,greater_is_better=False)

tpot = TPOTRegressor(random_state=1,verbosity=3, scoring = rmsle_loss,periodic_checkpoint_folder='C:\\Users\\jstnjc\\Google Drive\\Kaggle\\Advanced House Prices\\NoEng200', population_size=300,offspring_size= 400,max_eval_time_mins=10, use_dask=True)
tpot.fit(np.array(X_train),np.array(y_train))