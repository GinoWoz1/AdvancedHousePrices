# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 12:05:00 2018

@author: jjonus
"""

#TPOT test
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor,RandomForestClassifier # Regressor and Classifiers
from sklearn.model_selection import learning_curve,validation_curve,cross_val_score,GridSearchCV,train_test_split # validation
from sklearn.feature_selection import  mutual_info_regression # mutual information
from treeinterpreter import treeinterpreter as ti # tree interpretations
from scipy.cluster import hierarchy as hc # used for dendrogram analysis
import matplotlib.pyplot as plt # plotting
from fastai.structured import * # get_sample function as well 
from fancyimpute import MICE # imputation
from fastai.imports import *  # fast ai library
from pandas_summary import * 
from sklearn import metrics # for metrics on Grid Search CV
import pandas as pd # pandas
import seaborn as sns # seaborn plotting
import numpy as np # np library
from scipy import stats 
import missingno as msno # view missing data
from rfpimp import *
import warnings
import sys
import traceback
import time
import config    
import itertools # create combo list
import os
import tpot.metrics

# modeling libraries
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve,validation_curve,cross_val_score,GridSearchCV,train_test_split,RepeatedKFold
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from statsmodels.tools.tools import add_constant


warnings.filterwarnings('ignore')
%matplotlib inline

url = 'https://github.com/GinoWoz1/AdvancedHousePrices/raw/master/'
    
df_train = pd.read_csv(url + 'train.csv')
df_test = pd.read_csv(url +'test.csv')

df_train.columns.to_series().groupby(df_train.dtypes).groups

# Split data set into test and train

ntrain = df_train.shape[0]
ntest = df_test.shape[0]
y_train = df_train.SalePrice.values
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

# see missing data

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)

# another visualization of the missing data

missing_data = all_data.columns[all_data.isnull().any()].tolist()
msno.matrix(all_data[missing_data])

# analyze correlation between nullity 

msno.heatmap(all_data[missing_data],figsize = (20,20))

# fill in missing data

all_data['PoolQC'] = all_data['PoolQC'].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

#fill in missing data for garage, basement and mason area

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')

for col in ( 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
    
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
    
for col in ('BsmtHalfBath', 'BsmtFullBath', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1','BsmtFinSF2'):
    all_data[col] = all_data[col].fillna(0)


all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data["GarageYrBlt"] = all_data.groupby("Neighborhood")["GarageYrBlt"].transform( lambda x: x.fillna(x.mean()))

# fill in MSzoning values based on knowlege of data
    
all_data.loc[2216,'MSZoning'] = 'RM'
all_data.loc[2904,'MSZoning'] = 'RM'
all_data.loc[1915,'MSZoning'] = 'RM'
all_data.loc[2250,'MSZoning'] = 'RM'
all_data.loc[1915,'Utilities'] = 'AllPub'
all_data.loc[1945,'Utilities'] = 'AllPub'
all_data.loc[2473,'Functional'] = 'Typ'
all_data.loc[2216,'Functional'] = 'Typ'
all_data.loc[2489,'SaleType'] = 'WD'
all_data.loc[2151,'Exterior1st'] = 'Plywood'
all_data.loc[2151,'Exterior2nd'] = 'Plywood'
all_data.loc[1379,'Electrical'] = 'SBrkr'
# analyze which neighborhoods have null data

all_data[pd.isna(all_data['LotFrontage'])].groupby("Neighborhood").size().sort_values(ascending = False)

# change from number to categorical variable (set in string)

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['HalfBathTransformed'] = all_data['HalfBath'] * 0.5
all_data['HalfBathBsmtTransformed'] = all_data['BsmtHalfBath'] * 0.5
all_data['Baths'] = all_data['FullBath'] + all_data['HalfBathTransformed'] + all_data['HalfBathBsmtTransformed']  + all_data['BsmtFullBath']

# variable selection - create neighborhood, Subclass and quality groupings

#Excellent, Good, Typical, Fair, Poor, None: Convert to 0-5 scale

cols_ExGd = ['ExterQual','ExterCond','BsmtQual','BsmtCond',
             'HeatingQC','KitchenQual','FireplaceQu','GarageQual',
            'GarageCond','PoolQC']

dict_ExGd = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0}

for col in cols_ExGd:
    all_data[col].replace(dict_ExGd, inplace=True)

all_data.loc[1555,'KitchenQual'] = 3

# create new neighborhood groups

def neigh(row):
    if row['Neighborhood'] == 'Blmngtn':
        return 'BNS'
    elif row['Neighborhood'] =='Gilbert':
        return 'BNS'
    elif row['Neighborhood'] =='NWAmes':
        return 'BNS' 
    elif row['Neighborhood'] =='SawyerW':
        return 'BNS'
    elif row['Neighborhood']=='Blueste':
        return 'BlueSwis'
    elif row['Neighborhood']=='Mitchel':
        return 'BlueSwis' 
    elif row['Neighborhood'] =='NAmes':
        return 'BlueSwis' 
    elif row['Neighborhood'] =='NPkVill':
        return 'BlueSwis' 
    elif row['Neighborhood'] =='Sawyer':
        return 'BlueSwis' 
    elif row['Neighborhood']=='SWISU':
        return 'BlueSwis'
    elif row['Neighborhood']=='BrDale':
        return 'BrMeadow' 
    elif row['Neighborhood'] =='IDOTRR':
        return 'BrMeadow' 
    elif row['Neighborhood'] =='MeadowV':
        return 'BrMeadow' 
    elif row['Neighborhood'] =='BrkSide':
        return 'BEO' 
    elif row['Neighborhood']=='Edwards':
        return 'BEO'
    elif row['Neighborhood']=='OldTown':
        return 'BEO'
    elif row['Neighborhood'] =='NoRidge':
        return 'NNS' 
    elif row['Neighborhood'] =='NridgHt':
        return 'NNS' 
    elif row['Neighborhood'] =='StoneBr':
        return 'NNS' 
    elif row['Neighborhood']=='Somerst':
        return 'STV'
    elif row['Neighborhood']=='Timber':
        return 'STV' 
    elif row['Neighborhood']=='Veenker':
        return 'STV'
    elif row['Neighborhood']=='CollgCr':
        return 'C3'
    elif row['Neighborhood']=='Crawfor':
        return 'C3' 
    elif row['Neighborhood']=='ClearCr':
        return 'C3'
    else:
        return row['Neighborhood']

all_data['NeighGroup'] = all_data.apply(neigh,axis=1)  

# create column for year built after a certain date

all_data['After91'] = np.where(all_data['YearBuilt'] > 1991,1,0)  
all_data['YearBuiltRev'] = all_data['YearBuilt'] - 1875

# Exter and KitchenQual Combined

all_data['KitchenExterQual'] = all_data['KitchenQual'] + all_data['ExterQual']

dummy_cols = ['2ndFlrSF',
              'MiscVal','ScreenPorch','WoodDeckSF','OpenPorchSF',
              'EnclosedPorch','MasVnrArea','GarageArea','Fireplaces','TotalBsmtSF']

for col in dummy_cols:
    all_data['Has'+col] = (all_data[col]>0).astype(int)
    
# column for MasVnrStone    
all_data['HasVnrStone'] = np.where(all_data['MasVnrType'] == 'Stone',1,0)

#columns for HeatingQC
all_data['HeatingEx'] = np.where(all_data['HeatingQC'] == 5,1,0)

# duplex
all_data['IsDuplex'] = np.where(all_data['BldgType'] == "Duplex",1,0)

# foundation is concrete
all_data['HasPconc'] = np.where(all_data['Foundation'] == "PConc",1,0)

# evaluation metrics and model vetting function

def print_score(m):
    res = [rmse(m.predict(X_train),y_train),rmse(m.predict(X_valid),y_valid),m.score(X_train,y_train),m.score(X_valid,y_valid)]
    if hasattr(m,'oob_score_'): res.append(m.oob_score_)
    print(res)

train = all_data[:ntrain]
test = all_data[ntrain:]
y_train = df_train.SalePrice.values

numeric_data_train = train.select_dtypes(include = ['float64','int64'])
solver=MICE()
Imputed_dataframe_train= pd.DataFrame(data = solver.complete(numeric_data_train),columns = numeric_data_train.columns,index = numeric_data_train.index)
numeric_data_test = test.select_dtypes(include = ['float64','int64'])
solver=MICE()
Imputed_dataframe_test= pd.DataFrame(data = solver.complete(numeric_data_test),columns = numeric_data_test.columns,index = numeric_data_test.index)

train['LotFrontage'] = Imputed_dataframe_train['LotFrontage']
test['LotFrontage'] = Imputed_dataframe_test['LotFrontage']

X_train,X_valid,y_train,y_valid = train_test_split(train,y_train,test_size=0.25,random_state=1)

X_train= pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(test)

# test TPOT
import xgboost as xgb
from xgboost import XGBRegressor
from tpot import TPOTRegressor

X_train = X_train[['TotalBsmtSF','KitchenExterQual','GarageArea','GarageCars','GrLivArea','KitchenQual','Baths','BsmtQual','ExterQual','FullBath','GarageYrBlt','YearBuilt','YearBuiltRev','1stFlrSF','LotFrontage','FireplaceQu','YearRemodAdd','Foundation_PConc','HasPconc','TotRmsAbvGrd','Fireplaces','HeatingQC','BsmtFinSF1','HasFireplaces','GarageFinish_Unf','HasOpenPorchSF','NeighGroup_NNS','GarageFinish_Fin','OverallCond','GarageType_Attchd','BsmtUnfSF','NeighGroup_BlueSwis','HeatingEx','GarageType_Detchd','MSSubClass_60','Foundation_CBlock','2ndFlrSF','GarageQual','Exterior2nd_VinylSd','NeighGroup_BEO','GarageCond','Exterior1st_VinylSd','BsmtFinType1_GLQ','Neighborhood_NAmes','SaleCondition_Partial','LotShape_Reg','SaleType_New','Neighborhood_NridgHt','HasWoodDeckSF','HasMasVnrArea','MasVnrType_None','LotArea','NeighGroup_STV','GarageFinish_RFn','MSSubClass_30','MasVnrArea','MSZoning_RM','CentralAir_N','NeighGroup_BNS','HalfBathTransformed','HalfBath','HasGarageArea','Electrical_SBrkr','WoodDeckSF','GarageFinish_None','LotShape_IR1','NeighGroup_C3','BsmtCond','MSZoning_RL','GarageType_None','CentralAir_Y','Neighborhood_CollgCr','Neighborhood_NoRidge','Neighborhood_Gilbert','HasVnrStone','MasVnrType_Stone','NeighGroup_BrMeadow','BedroomAbvGr','Electrical_FuseA','Foundation_BrkTil','HouseStyle_2Story','BsmtExposure_Gd','Exterior1st_Wd Sdng','MasVnrType_BrkFace','Neighborhood_OldTown','PavedDrive_N','RoofStyle_Gable','Fence_MnPrv','Fence_None','Neighborhood_BrkSide','BsmtFinType1_ALQ']]
X_valid = X_valid[['TotalBsmtSF','KitchenExterQual','GarageArea','GarageCars','GrLivArea','KitchenQual','Baths','BsmtQual','ExterQual','FullBath','GarageYrBlt','YearBuilt','YearBuiltRev','1stFlrSF','LotFrontage','FireplaceQu','YearRemodAdd','Foundation_PConc','HasPconc','TotRmsAbvGrd','Fireplaces','HeatingQC','BsmtFinSF1','HasFireplaces','GarageFinish_Unf','HasOpenPorchSF','NeighGroup_NNS','GarageFinish_Fin','OverallCond','GarageType_Attchd','BsmtUnfSF','NeighGroup_BlueSwis','HeatingEx','GarageType_Detchd','MSSubClass_60','Foundation_CBlock','2ndFlrSF','GarageQual','Exterior2nd_VinylSd','NeighGroup_BEO','GarageCond','Exterior1st_VinylSd','BsmtFinType1_GLQ','Neighborhood_NAmes','SaleCondition_Partial','LotShape_Reg','SaleType_New','Neighborhood_NridgHt','HasWoodDeckSF','HasMasVnrArea','MasVnrType_None','LotArea','NeighGroup_STV','GarageFinish_RFn','MSSubClass_30','MasVnrArea','MSZoning_RM','CentralAir_N','NeighGroup_BNS','HalfBathTransformed','HalfBath','HasGarageArea','Electrical_SBrkr','WoodDeckSF','GarageFinish_None','LotShape_IR1','NeighGroup_C3','BsmtCond','MSZoning_RL','GarageType_None','CentralAir_Y','Neighborhood_CollgCr','Neighborhood_NoRidge','Neighborhood_Gilbert','HasVnrStone','MasVnrType_Stone','NeighGroup_BrMeadow','BedroomAbvGr','Electrical_FuseA','Foundation_BrkTil','HouseStyle_2Story','BsmtExposure_Gd','Exterior1st_Wd Sdng','MasVnrType_BrkFace','Neighborhood_OldTown','PavedDrive_N','RoofStyle_Gable','Fence_MnPrv','Fence_None','Neighborhood_BrkSide','BsmtFinType1_ALQ']]
X_test = X_test[['TotalBsmtSF','KitchenExterQual','GarageArea','GarageCars','GrLivArea','KitchenQual','Baths','BsmtQual','ExterQual','FullBath','GarageYrBlt','YearBuilt','YearBuiltRev','1stFlrSF','LotFrontage','FireplaceQu','YearRemodAdd','Foundation_PConc','HasPconc','TotRmsAbvGrd','Fireplaces','HeatingQC','BsmtFinSF1','HasFireplaces','GarageFinish_Unf','HasOpenPorchSF','NeighGroup_NNS','GarageFinish_Fin','OverallCond','GarageType_Attchd','BsmtUnfSF','NeighGroup_BlueSwis','HeatingEx','GarageType_Detchd','MSSubClass_60','Foundation_CBlock','2ndFlrSF','GarageQual','Exterior2nd_VinylSd','NeighGroup_BEO','GarageCond','Exterior1st_VinylSd','BsmtFinType1_GLQ','Neighborhood_NAmes','SaleCondition_Partial','LotShape_Reg','SaleType_New','Neighborhood_NridgHt','HasWoodDeckSF','HasMasVnrArea','MasVnrType_None','LotArea','NeighGroup_STV','GarageFinish_RFn','MSSubClass_30','MasVnrArea','MSZoning_RM','CentralAir_N','NeighGroup_BNS','HalfBathTransformed','HalfBath','HasGarageArea','Electrical_SBrkr','WoodDeckSF','GarageFinish_None','LotShape_IR1','NeighGroup_C3','BsmtCond','MSZoning_RL','GarageType_None','CentralAir_Y','Neighborhood_CollgCr','Neighborhood_NoRidge','Neighborhood_Gilbert','HasVnrStone','MasVnrType_Stone','NeighGroup_BrMeadow','BedroomAbvGr','Electrical_FuseA','Foundation_BrkTil','HouseStyle_2Story','BsmtExposure_Gd','Exterior1st_Wd Sdng','MasVnrType_BrkFace','Neighborhood_OldTown','PavedDrive_N','RoofStyle_Gable','Fence_MnPrv','Fence_None','Neighborhood_BrkSide','BsmtFinType1_ALQ']]


def rmsle_loss(y_true, y_pred):
	assert len(y_true) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y_true[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y_true))) ** 0.5

rmsle_loss = make_scorer(rmsle_loss,greater_is_better=False)

tpot = TPOTRegressor(verbosity=3,scoring=rmsle_loss,population_size=200,periodic_checkpoint_folder='C:\\Users\\jjonus\\Google Drive\\Kaggle\\Advanced House Prices')
tpot.fit(X_train,y_train)
print(tpot.score(X_valid,y_valid))
tpot.export('houses_pipeline.py')

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.svm import LinearSVR
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

exported_pipeline = make_pipeline(
    MinMaxScaler(),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    GradientBoostingRegressor(alpha=0.75, learning_rate=0.1, loss="ls", max_depth=4, max_features=0.6000000000000001, min_samples_leaf=2, min_samples_split=2, n_estimators=100, subsample=0.8500000000000001)
)

exported_pipeline.fit(X_train,y_train)
print_score(exported_pipeline)

y_pred = exported_pipeline.predict(X_test)
df_test['SalePrice'] = y_pred
submissions = df_test[['Id','SalePrice']]
submissions['Id'] = submissions['Id'].astype(int)
submissions.to_csv(current_wd + '\\lr_submissions.csv',encoding='utf-8',index=False)

# highest score .142 - highest so far in my attempts. Try with correlated predictors removed. 

X_train.drop(['HasFireplaces'],axis=1,inplace=True)
X_test.drop(['HasFireplaces'],axis=1,inplace=True)
X_valid.drop(['HasFireplaces'],axis=1,inplace=True)
X_train.drop(['HalfBath','HalfBathTransformed'],axis=1,inplace=True)
X_test.drop(['HalfBath','HalfBathTransformed'],axis=1,inplace=True)
X_valid.drop(['HalfBath','HalfBathTransformed'],axis=1,inplace=True)
X_train.drop(['Exterior2nd_VinylSd'],axis=1,inplace=True)
X_test.drop(['Exterior2nd_VinylSd'],axis=1,inplace=True)
X_valid.drop(['Exterior2nd_VinylSd'],axis=1,inplace=True)
X_train.drop(['HeatingEx'],axis=1,inplace=True)
X_test.drop(['HeatingEx'],axis=1,inplace=True)
X_valid.drop(['HeatingEx'],axis=1,inplace=True)
X_train.drop(['HasMasVnrArea','MasVnrType_Stone'],axis=1,inplace=True)
X_test.drop(['HasMasVnrArea','MasVnrType_Stone'],axis=1,inplace=True)
X_valid.drop(['HasMasVnrArea','MasVnrType_Stone'],axis=1,inplace=True)
X_train.drop(['HasWoodDeckSF'],axis=1,inplace=True)
X_test.drop(['HasWoodDeckSF'],axis=1,inplace=True)
X_valid.drop(['HasWoodDeckSF'],axis=1,inplace=True)
X_train.drop(['GarageYrBlt','YearBuiltRev'],axis=1,inplace=True)
X_test.drop(['GarageYrBlt','YearBuiltRev'],axis=1,inplace=True)
X_valid.drop(['GarageYrBlt','YearBuiltRev'],axis=1,inplace=True)
X_train.drop(['GarageFinish_None','GarageType_None'],axis=1,inplace=True)
X_test.drop(['GarageFinish_None','GarageType_None'],axis=1,inplace=True)
X_valid.drop(['GarageFinish_None','GarageType_None'],axis=1,inplace=True)
X_train.drop(['HasPconc'],axis=1,inplace=True)
X_test.drop(['HasPconc'],axis=1,inplace=True)
X_valid.drop(['HasPconc'],axis=1,inplace=True)
X_train.drop(['HasGarageArea'],axis=1,inplace=True)
X_test.drop(['HasGarageArea'],axis=1,inplace=True)
X_valid.drop(['HasGarageArea'],axis=1,inplace=True)
