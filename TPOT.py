# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 12:05:00 2018

@author: jjonus
"""

#TPOT scripts
from jfunc import ind_corr,tpot_advhouse,rmsle,pear_corr,drop_cols,TrainRegress
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

warnings.filterwarnings('ignore')

# load in data from github

X_train,X_valid,X_test,df_test,df_train,y_train,y_valid = tpot_advhouse(plot = False)

# look at correlations
pear_corr(X_train)

# remove cols

#remove = ['Foundation_PConc','FoudnationGroup_Group1','NeighGroup_BrMeadow','Foundation_CBlock','CarageCond','GarageType_Detchd','NeighGroup_BEO','HouseStyle_2Story','Neighborhood_CollgCr','MSSubClassGroup_60','MasVnrType_Stone','HasPconc','Electrical_SBrkr','HasVnrStone','LotShape_IR1','MasVnrType_None','LotShape_Reg','KitchenQual','Neighborhood_Gilbert','HasGarageArea','Neighborhood_NWAmes','ExposureGroup_None','MSZoning_RL','MSSubClass_30','MSZoning_RM','ELectrical_FuseA','Neighborhood_NAmes','ElectricalGroup_Group2']
remove = ['BsmtCond_None','BsmtFinType_None','BsmtQual_None','GarageType_None','GarageFinish_None','GarageCond_None','SaleType_New','BsmtFinType2_None','BsmtFinType1_None','Exterior1st_VinylSd']

X_train = drop_cols(X_train,remove)
X_valid = drop_cols(X_valid,remove)
X_test = drop_cols(X_test,remove)
                                
# scoring function for validation data and tpot run

def print_score(m):
    res = [rmsle(m.predict(X_train),y_train),rmsle(m.predict(X_valid),y_valid),m.score(X_train,y_train),m.score(X_valid,y_valid)]
    if hasattr(m,'oob_score_'): res.append(m.oob_score_)
    print(res) 
    
def rmsle_tpot(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y_true[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    if not (y_true >= 0).all() and not (y_pred >= 0).all():
            raise ValueError("Mean Squared Logarithmic Error cannot be used when "
                             "targets contain negative values.")
    return (sum(terms_to_sum) * (1.0/len(y_true))) ** 0.5

rmsle_tpot = make_scorer(rmsle_tpot,greater_is_better=False)   


# set k fold parameters

rkfold = RepeatedKFold(n_splits =10,n_repeats=5)

# run tpot

tpot = TPOTRegressor(verbosity=3,scoring=rmsle_tpot,population_size=300,periodic_checkpoint_folder='C:\\Users\\jjonus\\Google Drive\\Kaggle\\Advanced House Prices\\Eng200',max_eval_time_mins=5,n_jobs=1,cv=rkfold)
tpot.fit(X_train,y_train)

# build scorer for custom train regress function

def rmsle_cv(y_true, y_pred) : 
    assert len(y_true) == len(y_pred)
    if not (y_true >= 0).all() and not (y_pred >= 0).all():
        raise ValueError("Mean Squared Logarithmic Error cannot be used when "
                         "targets contain negative values.")
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_true))**2))

rmsle_cv = make_scorer(rmsle_cv,greater_is_better=False)   

# create pipeline and fit
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RidgeCV()),
    ExtraTreesRegressor(bootstrap=True, max_features=0.55, min_samples_leaf=1, min_samples_split=7, n_estimators=1000)
)

exported_pipeline.fit(X_train,y_train)

#gather the outlier data
model,cv_score,grid_results,df_outliers = TrainRegress(exported_pipeline,sigma=3,scorer = rmsle_cv, X = X_train,y=y_train)   

def error_calc(y_true, y_pred): 
    assert len(y_true) == len(y_pred)
    return np.sqrt((np.log(1+y_pred) - np.log(1+y_true))**2)

df_outliers['Rmsle'] = error_calc(df_outliers['Target'],df_outliers['Prediction'])

# send predictions to csv

from jfunc import directory

current_wd = directory()

y_pred = exported_pipeline.predict(X_test)

df_test['SalePrice'] = y_pred
submissions = df_test[['Id','SalePrice']]
submissions['Id'] = submissions['Id'].astype(int)
submissions.to_csv(current_wd + '\\lr_submissions.csv',encoding='utf-8',index=False)




