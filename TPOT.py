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

mingw_path = 'C:\\Users\\jstnj\\Anaconda3\\Library\\mingw-w64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

warnings.filterwarnings('ignore')

# load in data from github

X_train,X_valid,X_test,df_test,df_train,y_train,y_valid,outliers = tpot_advhouse(plot = False,remove_outliers=True)

# look at correlations

pear_corr(X_train)

# remove cols

remove = ['BldgType_Duplex','KitchenAbvGr','HalfBath','HalfBathTransoformed','After91','FullBath','YearBuilt','Foundation_PConc','SaleType_New','MasVnrType_Stone','BsmtFinType2_None','KitchenQual','HeatingEx','FireplaceQu','WoodDeckSF','MasVnrArea','OpenPorchSF','GarageYrBlt','GarageArea','TotalBsmtSF','HalfBathTransformed','ExterQual']

X_train = drop_cols(X_train,remove)
X_valid = drop_cols(X_valid,remove)
X_test = drop_cols(X_test,remove)

from jfunc import df_datatypes,cardinality,define_vars

df_datatypes(X_train)
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

exported_pipeline.fit(X_train,y_train)

#gather the outlier data
model,cv_score,grid_results,df_outliers = TrainRegress(exported_pipeline,sigma=3,scorer = rmsle_cv, X = X_valid,y=y_valid)   

def error_calc(y_true, y_pred): 
    assert len(y_true) == len(y_pred)
    return np.sqrt((np.log(1+y_pred) - np.log(1+y_true))**2)

df_outliers['Rmsle'] = error_calc(df_outliers['Target'],df_outliers['Prediction'])

# send predictions to csv

y_pred = exported_pipeline.predict(X_test)

df_test['SalePrice'] = y_pred
submissions = df_test[['Id','SalePrice']]
submissions['Id'] = submissions['Id'].astype(int)
submissions.to_csv(current_wd + '\\lr_submissions.csv',encoding='utf-8',index=False)


exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RidgeCV()),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.05, min_samples_leaf=9, min_samples_split=3, n_estimators=100)),
    GradientBoostingRegressor(alpha=0.85, learning_rate=0.1, loss="ls", max_depth=10, max_features=0.5, min_samples_leaf=18, min_samples_split=13, n_estimators=100, subsample=0.8)
)


# test olivier feature importance method

# test further columns removal using olivier feature importance method

import time
from lightgbm import LGBMRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
%matplotlib inline

import warnings
warnings.simplefilter('ignore', UserWarning)

import gc
gc.enable()

from sklearn.ensemble import GradientBoostingRegressor
exported_pipeline = GradientBoostingRegressor(alpha=0.99,   learning_rate=0.1, loss="ls", max_depth=9, max_features=0.15, min_samples_leaf=10, min_samples_split=8, n_estimators=100, subsample=0.9)


def get_feature_importances(data,y, shuffle, seed=None):
    # Gather real features
    train_features = X_train
    # Go over fold and keep track of CV score (train and valid) and feature importances
    
    # Shuffle target if required
    y = y.copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = y.copy().sample(frac=1.0)
    
    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest

    exported_pipeline.fit(X_train,y_train)
    # Fit the model
    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance"] = exported_pipeline.feature_importances_
    
    return imp_df

np.random.seed(123)

# build actual importance list

actual_imp_df = pd.DataFrame()

nb_runs = 1000
import time
start = time.time()
dsp = ''
for i in range(nb_runs):
    # Get current run importances
    imp_df = get_feature_importances(data=X_train,y=y_train, shuffle=False)
    imp_df['run'] = i + 1 
    # Concat the latest importances with the old ones
    actual_imp_df = pd.concat([actual_imp_df, imp_df], axis=0)
    # Erase previous message
    for l in range(len(dsp)):
        print('\b', end='', flush=True)
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='', flush=True)

# null importances

null_imp_df = pd.DataFrame()
nb_runs = 1000
import time
start = time.time()
dsp = ''
for i in range(nb_runs):
    # Get current run importances
    imp_df = get_feature_importances(data=X_train,y=y_train, shuffle=True)
    imp_df['run'] = i + 1 
    # Concat the latest importances with the old ones
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    # Erase previous message
    for l in range(len(dsp)):
        print('\b', end='', flush=True)
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='', flush=True)

null_imp_df = null_imp_df[['feature','importance']]
   
def display_distributions(actual_imp_df_, null_imp_df_, feature_):
    plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 1)
    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_].values, label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_].mean(), 
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Split Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (split) Distribution for %s ' % feature_.upper())

for x in X_train.columns.tolist():
    display_distributions(actual_imp_df,null_imp_df,x)



plt.figure(figsize=(13, 6))
gs = gridspec.GridSpec(1, 1)
# Plot Split importances
ax = plt.subplot(gs[0, 0])
a =ax.hist(null_imp_df.loc[null_imp_df['feature'] == 'IsDuplex'].values[:,1])
    