# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 12:05:00 2018

@author: jjonus
"""

#TPOT scripts
from jfunc import ind_corr,tpot_advhouse,rmsle,pear_corr,drop_cols,TrainRegress,directory
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import train_test_split,RepeatedKFold,GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from tpot.builtins import StackingEstimator
from sklearn.linear_model import RidgeCV
from sklearn.metrics import make_scorer
from tpot import TPOTRegressor  
import pandas as pd
import numpy as np
import warnings
import math
import os

current_wd = directory()

mingw_path = current_wd+ '\\Anaconda3\\Library\\mingw-w64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

warnings.filterwarnings('ignore')

# load in data from github

X_train,X_valid,X_test,df_test,df_train,y_train,y_valid,outliers = tpot_advhouse(plot = False,remove_outliers=True)

# look at correlations

pear_corr(X_train)

# examine 0s

from jfunc import num_zeros

# return columns with more than 95% zeros

df_zero,columns = num_zeros(X_train,0.95)

# remove cols

remove = columns
X_train = drop_cols(X_train,remove)
X_valid = drop_cols(X_valid,remove)
X_test = drop_cols(X_test,remove)

from jfunc import df_datatypes,cardinality,define_vars

# scoring function for validation data and tpot run

def print_score(m):
    res = [rmsle(m.predict(X_train),y_train),rmsle(m.predict(X_valid),y_valid),m.score(X_train,y_train),m.score(X_valid,y_valid)]
    if hasattr(m,'oob_score_'): res.append(m.oob_score_)
    print(res) 
    
def rmsle_tpot(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    try:
        terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y_true[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    except:
        return float('inf')
    if not (y_true >= 0).all() and not (y_pred >= 0).all():
            return float('inf')
    return (sum(terms_to_sum) * (1.0/len(y_true))) ** 0.5

rmsle_tpot = make_scorer(rmsle_tpot,greater_is_better=False)   

# set k fold parameters

rkfold = RepeatedKFold(n_splits =10,n_repeats=5)

# run tpot
            
tpot = TPOTRegressor(verbosity=3,population_size=300,offspring_size=400,scoring=rmsle_tpot,periodic_checkpoint_folder=current_wd + '\\Kaggle\\Advanced House Prices\\Eng200',max_eval_time_mins=5,n_jobs=-1,cv=rkfold,use_dask=True)
tpot.fit(X_train,y_train)

# build scorer for custom train regress function

def rmsle_cv(y_true, y_pred) : 
    assert len(y_true) == len(y_pred)
    if not (y_true >= 0).all() and not (y_pred >= 0).all():
        raise ValueError("Mean Squared Logarithmic Error cannot be used when "
                         "targets contain negative values.")
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_true))**2))

rmsle_cv = make_scorer(rmsle_cv,greater_is_better=False)   

# score models

opt_models = dict()
score_models = pd.DataFrame(columns=['mean','std'])
model = 'MinMaxLassExtraExtra'

exported_pipeline = make_pipeline(
    MinMaxScaler(),
    StackingEstimator(estimator=LassoLarsCV(normalize=False)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.45, min_samples_leaf=8, min_samples_split=15, n_estimators=100)),
    ExtraTreesRegressor(bootstrap=False, max_features=0.6000000000000001, min_samples_leaf=1, min_samples_split=11, n_estimators=100)
)

exported_pipeline.fit(X_train,y_train)
print_score(exported_pipeline)

opt_models[model],cv_score,grid_results,df_outliers = TrainRegress(exported_pipeline,sigma=3,scorer = rmsle_cv, X = X_train,y=y_train,splits=10)   

cv_score.name=model
score_models = score_models.append(cv_score)

# send predictions to csv

y_pred = exported_pipeline.predict(X_test)

df_test['SalePrice'] = y_pred
submissions = df_test[['Id','SalePrice']]
submissions['Id'] = submissions['Id'].astype(int)
submissions.to_csv(current_wd + '\\lr_submissions.csv',encoding='utf-8',index=False)





    