# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 12:05:00 2018

@author: jjonus
"""

#TPOT test

from jfunc import ind_corr,tpot_advhouse,rmsle_loss,rmsle
from xgboost import XGBRegressor
from tpot import TPOTRegressor  
import xgboost as xgb 
import warnings
import os

warnings.filterwarnings('ignore')

# load scorer

def print_score(m,columns):
    res = [rmsle(m.predict(X_train[columns]),y_train),rmsle(m.predict(X_valid[columns]),y_valid),m.score(X_train[columns],y_train),m.score(X_valid[columns],y_valid)]
    if hasattr(m,'oob_score_'): res.append(m.oob_score_)
    print(res)
    
# load in data from jfunc file

X_train,X_valid,X_test,df_test,df_train,y_train,y_valid = tpot_advhouse(plot = False)

new_df, columns = ind_corr(X_train,y_train,0.20,-0.20)

"""
plt.spy(X_train,markersize=5)
"""
# test TPOT
    
tpot = TPOTRegressor(verbosity=3,scoring=rmsle_loss,generations = 50,population_size=50,offspring_size= 50,periodic_checkpoint_folder='C:\\Users\\jjonus\\Google Drive\\Kaggle\\Advanced House Prices\\Eng200',max_eval_time_mins=10,warm_start=True)
tpot.fit(X_train[columns],y_train)

"""
print(tpot.score(X_valid,y_valid))
tpot.export('houses_pipeline.py')
"""

# Parameter tuning on GB/Lasso model
    
current_loc = os.getcwd()
if 'jjonus' in current_loc:
    current_wd = ('C:\\Users\\jjonus\\Google Drive\\Kaggle\\Advanced House Prices')
elif 'jstnj' in current_loc:
    current_wd = ('C:\\Users\\jstnj\\Google Drive\\Kaggle\\Advanced House Prices')

y_pred = exported_pipeline.predict(X_test[columns])
df_test['SalePrice'] = y_pred
submissions = df_test[['Id','SalePrice']]
submissions['Id'] = submissions['Id'].astype(int)
submissions.to_csv(current_wd + '\\lr_submissions.csv',encoding='utf-8',index=False)


# testing randomized search 
alpha_range = np.arange(0.6,0.9,0.05)
learning_rate = np.arange(0.05,0.3,0.01)
max_depth = np.arange(5,15,1)
max_features = np.arange(0.40,0.80,0.05)
min_samples_leaf = np.arange(3,7,1)
min_samples_split = np.arange (2,7,1)
subsample = np.arange (0.2,0.8,0.05)
param = {"gradientboostingregressor__alpha":alpha_range,
         "gradientboostingregressor__learning_rate":learning_rate,
         "gradientboostingregressor__max_depth":max_depth,
         "gradientboostingregressor__max_features":max_features,
         "gradientboostingregressor__min_samples_leaf":min_samples_leaf,
         "gradientboostingregressor__subsample":subsample}
n_iter_search = 1000
random_search = RandomizedSearchCV(exported_pipeline,param_distributions=param,n_iter=n_iter_search,scoring=rmsle_loss)
random_search.fit(X_valid,y_valid)

