# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 12:05:00 2018

"""
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor
from tpot import TPOTRegressor  
import xgboost as xgb 
import warnings
import pandas as pd

warnings.filterwarnings('ignore')
    
# load in data

url = 'https://github.com/GinoWoz1/AdvancedHousePrices/raw/master/'
        
X_train = pd.read_csv(url + 'train_tpot_issue.csv')
y_train = pd.read_csv(url + 'y_train_tpot_issue.csv')

# loss function

def rmsle_loss(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y_true[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    if not (y_true >= 0).all() and not (y_pred >= 0).all():
            raise ValueError("Mean Squared Logarithmic Error cannot be used when "
                             "targets contain negative values.")
    return (sum(terms_to_sum) * (1.0/len(y_true))) ** 0.5

rmsle_loss = make_scorer(rmsle_loss,greater_is_better=False)   

# run tpot
    
tpot = TPOTRegressor(verbosity=3,scoring=rmsle_loss,generations = 50,population_size=50,offspring_size= 50,max_eval_time_mins=10,warm_start=True)
tpot.fit(X_train,y_train)



