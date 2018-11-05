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
import pickle as pck
import numpy as np

mingw_path = 'C:\\Users\\xiaotingjoyce\\Anaconda3\\Library\\mingw-w64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

warnings.filterwarnings('ignore')

"""
url = 'https://github.com/GinoWoz1/AdvancedHousePrices/raw/master/'

X_train = pd.read_csv(url + 'train_EngOutliers.csv',index_col="Unnamed: 0")

y_train = pd.read_csv(url + 'y_train_EngOutliers.csv',header=None)
"""

X_train = pd.read_csv('C:\\Users\\xiaotingjoyce\\Google Drive\\Kaggle\\GitAdvancedHousePrices\\train_EngOutliers.csv',index_col="Unnamed: 0")

y_train = pd.read_csv('C:\\Users\\xiaotingjoyce\\Google Drive\\Kaggle\\GitAdvancedHousePrices\\y_train_EngOutliers.csv',header=None,index_col=0)

def rmsle_loss(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    try:
        terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y_true[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    except:
        return float('inf')
    if not (y_true >= 0).all() and not (y_pred >= 0).all():
            return float('inf')
    return (sum(terms_to_sum) * (1.0/len(y_true))) ** 0.5

custom_config =  {

    'sklearn.linear_model.ElasticNetCV': {
        'l1_ratio': np.arange(0.0, 1.01, 0.05),
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    },

    'sklearn.ensemble.ExtraTreesRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.GradientBoostingRegressor': {
        'n_estimators': [100],
        'loss': ["ls", "lad", "huber", "quantile"],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05),
        'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    },

    'sklearn.ensemble.AdaBoostRegressor': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'loss': ["linear", "square", "exponential"],
    },

    'sklearn.tree.DecisionTreeRegressor': {
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    'sklearn.neighbors.KNeighborsRegressor': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    'sklearn.linear_model.LassoLarsCV': {
        'normalize': [True, False]
    },

    'sklearn.svm.LinearSVR': {
        'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
    },

    'sklearn.ensemble.RandomForestRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.linear_model.RidgeCV': {
    },

    'xgboost.XGBRegressor': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'nthread': [8]
    },

    # Preprocesssors
    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },

    'sklearn.preprocessing.MaxAbsScaler': {
    },

    'sklearn.preprocessing.MinMaxScaler': {
    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },

}
    
rmsle_loss = make_scorer(rmsle_loss,greater_is_better=False)

tpot = TPOTRegressor(verbosity=3, config_dict= custom_config,scoring = rmsle_loss,periodic_checkpoint_folder='C:\\Users\\xiaotingjoyce\\Google Drive\\Kaggle\\Advanced House Prices\\Eng_2', population_size=300,offspring_size= 400,max_eval_time_mins=10, use_dask=True)
tpot.fit(X_train,y_train[1])

eval_pareto = tpot.pareto_front_fitted_pipelines_
output2 = open('C:\\Users\\xiaotingjoyce\\Google Drive\\Kaggle\\Advanced House Prices\\Eng_2\\eval_pareto.pkl','wb')
pck.dump(eval_pareto,output2)











