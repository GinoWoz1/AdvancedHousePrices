# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 10:58:19 2018

@author: jjonus
"""
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
from jfunc import find_outliers

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
from math import ceil

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

def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def rmsle(y, y_pred): 
    assert len(y) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y))**2))

def print_score(m):
    res = [rmsle(m.predict(X_train),y_train),rmsle(m.predict(X_valid),y_valid),m.score(X_train,y_train),m.score(X_valid,y_valid)]
    if hasattr(m,'oob_score_'): res.append(m.oob_score_)
    print(res)

def dropcol_importances(rf, X_train, y_train):
    rf_ = clone(rf)
    rf_.random_state = 999
    rf_.fit(X_train, y_train)
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        rf_ = clone(rf)
        rf_.random_state = 999
        rf_.fit(X, y_train)
        o = rf_.oob_score_
        imp.append(baseline - o)
    imp = np.array(imp)
    I = pd.DataFrame(
            data={'Feature':X_train.columns,
                  'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    return I

# combination function
    
def list_trans(features):
    list_o_lists = []
    for c in range(1,len(features)+1):
        print(c)
        data =  list(itertools.combinations(features,c))
        list_o_lists.append(data)
    new_list = [[]]
    for i in list(list_o_lists):
        for c in i:
            new_list.append(c)
    new_list = [x for x in new_list if x]
    return new_list

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

y_valid = pd.Series(y_valid)
y_train = pd.Series(y_train)

X_test = test.copy()
X_train = pd.get_dummies(X_train).reset_index()

outliers = find_outliers(Ridge(),X_train,y_train)
X_train = X_train.drop(outliers)
y_train = y_train.drop(outliers)

X_test = pd.get_dummies(X_test)
X_valid = pd.get_dummies(X_valid)

sns.distplot(y_train)
sns.distplot(y_valid)
fig = plt.figure()

# gather mutual information from the data set

column_list = X_train.columns.tolist()

new_dict = {}

for c in column_list:
    Xmi_train = X_train[[c]]
    mi = mutual_info_regression(Xmi_train,y_train)
    new_dict[c] = sum(mi)

column_list = X_valid.columns.tolist()

new_dict2 = {}

for c in column_list:
    Xmi_valid = X_valid[[c]]
    mi = mutual_info_regression(Xmi_valid,y_valid)
    new_dict2[c] = sum(mi)

# test the structure of the trees
m = RandomForestRegressor(max_features = 0.5,n_estimators = 60 ,oob_score = True,min_samples_leaf=3)
m.fit(X_train,y_train)
print(m.oob_score_)

# examine tree
draw_tree(m.estimators_[0],X_train,precision=3)

# mutual information data
    
df_mi1 = pd.DataFrame.from_dict(new_dict,orient = 'index')
df_mi2 = pd.DataFrame.from_dict(new_dict2,orient= 'index')

df_mi1.to_csv(current_wd + '\\mi1.csv')
df_mi2.to_csv(current_wd + '\\mi2.csv')

 
X_train = X_train[['TotalBsmtSF','KitchenExterQual','GarageArea','GarageCars','GrLivArea','KitchenQual','Baths','BsmtQual','ExterQual','FullBath','GarageYrBlt','YearBuilt','YearBuiltRev','1stFlrSF','LotFrontage','FireplaceQu','YearRemodAdd','Foundation_PConc','HasPconc','TotRmsAbvGrd','Fireplaces','HeatingQC','BsmtFinSF1','HasFireplaces','GarageFinish_Unf','HasOpenPorchSF','NeighGroup_NNS','GarageFinish_Fin','OverallCond','GarageType_Attchd','BsmtUnfSF','NeighGroup_BlueSwis','HeatingEx','GarageType_Detchd','MSSubClass_60','Foundation_CBlock','2ndFlrSF','GarageQual','Exterior2nd_VinylSd','NeighGroup_BEO','GarageCond','Exterior1st_VinylSd','BsmtFinType1_GLQ','Neighborhood_NAmes','SaleCondition_Partial','LotShape_Reg','SaleType_New','Neighborhood_NridgHt','HasWoodDeckSF','HasMasVnrArea','MasVnrType_None','LotArea','NeighGroup_STV','GarageFinish_RFn','MSSubClass_30','MasVnrArea','MSZoning_RM','CentralAir_N','NeighGroup_BNS','HalfBathTransformed','HalfBath','HasGarageArea','Electrical_SBrkr','WoodDeckSF','GarageFinish_None','LotShape_IR1','NeighGroup_C3','BsmtCond','MSZoning_RL','GarageType_None','CentralAir_Y','Neighborhood_CollgCr','Neighborhood_NoRidge','Neighborhood_Gilbert','HasVnrStone','MasVnrType_Stone','NeighGroup_BrMeadow','BedroomAbvGr','Electrical_FuseA','Foundation_BrkTil','HouseStyle_2Story','BsmtExposure_Gd','Exterior1st_Wd Sdng','MasVnrType_BrkFace','Neighborhood_OldTown','PavedDrive_N','RoofStyle_Gable','Fence_MnPrv','Fence_None','Neighborhood_BrkSide','BsmtFinType1_ALQ']]
X_valid = X_valid[['TotalBsmtSF','KitchenExterQual','GarageArea','GarageCars','GrLivArea','KitchenQual','Baths','BsmtQual','ExterQual','FullBath','GarageYrBlt','YearBuilt','YearBuiltRev','1stFlrSF','LotFrontage','FireplaceQu','YearRemodAdd','Foundation_PConc','HasPconc','TotRmsAbvGrd','Fireplaces','HeatingQC','BsmtFinSF1','HasFireplaces','GarageFinish_Unf','HasOpenPorchSF','NeighGroup_NNS','GarageFinish_Fin','OverallCond','GarageType_Attchd','BsmtUnfSF','NeighGroup_BlueSwis','HeatingEx','GarageType_Detchd','MSSubClass_60','Foundation_CBlock','2ndFlrSF','GarageQual','Exterior2nd_VinylSd','NeighGroup_BEO','GarageCond','Exterior1st_VinylSd','BsmtFinType1_GLQ','Neighborhood_NAmes','SaleCondition_Partial','LotShape_Reg','SaleType_New','Neighborhood_NridgHt','HasWoodDeckSF','HasMasVnrArea','MasVnrType_None','LotArea','NeighGroup_STV','GarageFinish_RFn','MSSubClass_30','MasVnrArea','MSZoning_RM','CentralAir_N','NeighGroup_BNS','HalfBathTransformed','HalfBath','HasGarageArea','Electrical_SBrkr','WoodDeckSF','GarageFinish_None','LotShape_IR1','NeighGroup_C3','BsmtCond','MSZoning_RL','GarageType_None','CentralAir_Y','Neighborhood_CollgCr','Neighborhood_NoRidge','Neighborhood_Gilbert','HasVnrStone','MasVnrType_Stone','NeighGroup_BrMeadow','BedroomAbvGr','Electrical_FuseA','Foundation_BrkTil','HouseStyle_2Story','BsmtExposure_Gd','Exterior1st_Wd Sdng','MasVnrType_BrkFace','Neighborhood_OldTown','PavedDrive_N','RoofStyle_Gable','Fence_MnPrv','Fence_None','Neighborhood_BrkSide','BsmtFinType1_ALQ']]
X_test = X_test[['TotalBsmtSF','KitchenExterQual','GarageArea','GarageCars','GrLivArea','KitchenQual','Baths','BsmtQual','ExterQual','FullBath','GarageYrBlt','YearBuilt','YearBuiltRev','1stFlrSF','LotFrontage','FireplaceQu','YearRemodAdd','Foundation_PConc','HasPconc','TotRmsAbvGrd','Fireplaces','HeatingQC','BsmtFinSF1','HasFireplaces','GarageFinish_Unf','HasOpenPorchSF','NeighGroup_NNS','GarageFinish_Fin','OverallCond','GarageType_Attchd','BsmtUnfSF','NeighGroup_BlueSwis','HeatingEx','GarageType_Detchd','MSSubClass_60','Foundation_CBlock','2ndFlrSF','GarageQual','Exterior2nd_VinylSd','NeighGroup_BEO','GarageCond','Exterior1st_VinylSd','BsmtFinType1_GLQ','Neighborhood_NAmes','SaleCondition_Partial','LotShape_Reg','SaleType_New','Neighborhood_NridgHt','HasWoodDeckSF','HasMasVnrArea','MasVnrType_None','LotArea','NeighGroup_STV','GarageFinish_RFn','MSSubClass_30','MasVnrArea','MSZoning_RM','CentralAir_N','NeighGroup_BNS','HalfBathTransformed','HalfBath','HasGarageArea','Electrical_SBrkr','WoodDeckSF','GarageFinish_None','LotShape_IR1','NeighGroup_C3','BsmtCond','MSZoning_RL','GarageType_None','CentralAir_Y','Neighborhood_CollgCr','Neighborhood_NoRidge','Neighborhood_Gilbert','HasVnrStone','MasVnrType_Stone','NeighGroup_BrMeadow','BedroomAbvGr','Electrical_FuseA','Foundation_BrkTil','HouseStyle_2Story','BsmtExposure_Gd','Exterior1st_Wd Sdng','MasVnrType_BrkFace','Neighborhood_OldTown','PavedDrive_N','RoofStyle_Gable','Fence_MnPrv','Fence_None','Neighborhood_BrkSide','BsmtFinType1_ALQ']]

m = RandomForestRegressor(max_features = 0.5,n_estimators = 100 ,oob_score = True,min_samples_leaf=3)
m.fit(X_train,y_train)
print_score(m)

rmsle(X_train,y_train)
rmsle(X_valid,y_valid)

# check correlated variables
df_corr_mat = feature_corr_matrix(X_train)
df_corr_mat = df_corr_mat.dropna(axis='columns',how='all')
df_corr_mat = df_corr_mat.dropna()
df_corr_mat = df_corr_mat.values
corr_condensed = hc.distance.squareform(1-df_corr_mat)
z = hc.linkage(corr_condensed,method='average')
fig = plt.figure(figsize =(20,10))
dendrogram = hc.dendrogram(z,labels= X_train.columns,orientation = 'left',leaf_font_size =8)
plt.show()

# build correlation matrix for manual inspection

#matrix = feature_corr_matrix(X_train)

# Investigate correlation on no basement, duplex and MSSubClass 90

features = ['HasFireplaces','Fireplaces','FireplaceQu']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['HasFireplaces'],axis=1,inplace=True)
X_test.drop(['HasFireplaces'],axis=1,inplace=True)
X_valid.drop(['HasFireplaces'],axis=1,inplace=True)


m = RandomForestRegressor(max_features = 0.5,n_estimators = 50 ,oob_score = True,min_samples_leaf=3)
m.fit(X_train,y_train)
print_score(m)

# Saletype and condition


features = ['SaleType_New','SaleCondition_Partial']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['SaleType_New','SaleCondition_Partial'],axis=1,inplace=True)
X_test.drop(['SaleType_New','SaleCondition_Partial'],axis=1,inplace=True)
X_valid.drop(['SaleType_New','SaleCondition_Partial'],axis=1,inplace=True)


features = ['Exterior1st_VinylSd','Exterior2nd_VinylSd']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['HalfBath','HalfBathTransformed'],axis=1,inplace=True)
X_test.drop(['HalfBath','HalfBathTransformed'],axis=1,inplace=True)
X_valid.drop(['HalfBath','HalfBathTransformed'],axis=1,inplace=True)
X_train.drop(['Exterior2nd_VinylSd'],axis=1,inplace=True)
X_test.drop(['Exterior2nd_VinylSd'],axis=1,inplace=True)
X_valid.drop(['Exterior2nd_VinylSd'],axis=1,inplace=True)

m = RandomForestRegressor(max_features = 0.5,n_estimators = 59,oob_score = True,min_samples_leaf=3)
m.fit(X_train,y_train)
print_score(m)

# Heating

features = ['HeatingQC','HeatingEx']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['HeatingEx'],axis=1,inplace=True)
X_test.drop(['HeatingEx'],axis=1,inplace=True)
X_valid.drop(['HeatingEx'],axis=1,inplace=True)

#MasVnrArea

features = ['MasVnrArea','HasMasVnrArea','MasVnrType_Stone']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['HasMasVnrArea','MasVnrType_Stone'],axis=1,inplace=True)
X_test.drop(['HasMasVnrArea','MasVnrType_Stone'],axis=1,inplace=True)
X_valid.drop(['HasMasVnrArea','MasVnrType_Stone'],axis=1,inplace=True)

# Wood deck

features = ['WoodDeckSF','HasWoodDeckSF']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['HasWoodDeckSF'],axis=1,inplace=True)
X_test.drop(['HasWoodDeckSF'],axis=1,inplace=True)
X_valid.drop(['HasWoodDeckSF'],axis=1,inplace=True)

# YearBuilt

features = ['YearBuilt','YearBuiltRev','GarageYrBlt']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['GarageYrBlt','YearBuiltRev'],axis=1,inplace=True)
X_test.drop(['GarageYrBlt','YearBuiltRev'],axis=1,inplace=True)
X_valid.drop(['GarageYrBlt','YearBuiltRev'],axis=1,inplace=True)

# Garage,Finish type none

features = ['GarageType_None','GarageFinish_None']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['GarageFinish_None','GarageType_None'],axis=1,inplace=True)
X_test.drop(['GarageFinish_None','GarageType_None'],axis=1,inplace=True)
X_valid.drop(['GarageFinish_None','GarageType_None'],axis=1,inplace=True)

m = RandomForestRegressor(max_features = 0.5,n_estimators = 50 ,oob_score = True,min_samples_leaf=3)
m.fit(X_train,y_train)
print_score(m)

# Quality variables

features = ['KitchenQual','KitchenExterQual','ExterQual']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['KitchenQual'],axis=1,inplace=True)
X_test.drop(['KitchenQual'],axis=1,inplace=True)
X_valid.drop(['KitchenQual'],axis=1,inplace=True)

# examine next set of collinear variables

df = dropcol_importances(m,X_valid,y_valid)
I = importances(m, X_valid, y_valid,features = features)

# after this, the important created variable is scoring low but there is high collinearity with the 3rd variable of exterqual - need to examine.

features = ['KitchenExterQual','ExterQual']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

# more important variable is not the KitchenExterQual one but the other - correlation led to misinterpration

X_train.drop(['ExterQual'],axis=1,inplace=True)
X_test.drop(['ExterQual'],axis=1,inplace=True)
X_valid.drop(['ExterQual'],axis=1,inplace=True)

# dropcol again
df = dropcol_importances(m,X_valid,y_valid)

# GarageFinish_Unf Test negative collinearity as well
features = ['GarageFinish_Unf','GarageType_Detchd']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['GarageType_Detchd'],axis=1,inplace=True)
X_test.drop(['GarageType_Detchd'],axis=1,inplace=True)
X_valid.drop(['GarageType_Detchd'],axis=1,inplace=True)
# check Exterior1st Vinyld sd with correlations of .56 on a few variables

features = ['Exterior1st_VinylSd','YearBuilt','YearRemodAdd','Foundation_PConc','HasPconc','HeatingQC']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

# Garage
features = ['GarageQual','GarageCond','HasGarageArea']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['HasGarageArea'],axis=1,inplace=True)
X_test.drop(['HasGarageArea'],axis=1,inplace=True)
X_valid.drop(['HasGarageArea'],axis=1,inplace=True)

# Neighborhood_NridgeHt

features = ['Neighborhood_NridgHt','NeighGroup_NNS']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['Neighborhood_NridgHt'],axis=1,inplace=True)
X_test.drop(['Neighborhood_NridgHt'],axis=1,inplace=True)
X_valid.drop(['Neighborhood_NridgHt'],axis=1,inplace=True)

#MasVnrArea and Other correlated variable

features = ['MasVnrArea','MasVnrType_BrkFace']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['MasVnrType_BrkFace'],axis=1,inplace=True)
X_test.drop(['MasVnrType_BrkFace'],axis=1,inplace=True)
X_valid.drop(['MasVnrType_BrkFace'],axis=1,inplace=True)

# garageArea and cars

features = ['GarageArea','GarageCars']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

# Pconc

features = ['Foundation_PConc','HasPconc']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['HasPconc'],axis=1,inplace=True)
X_test.drop(['HasPconc'],axis=1,inplace=True)
X_valid.drop(['HasPconc'],axis=1,inplace=True)

# Oldtown

features = ['Neighborhood_OldTown','NeighGroup_BEO']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

#Bsmt  Qual

features = ['Fireplaces','FireplaceQU']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

#Rooms

features = ['TotRmsAbvGrd','BedroomAbvGr']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['BedroomAbvGr'],axis=1,inplace=True)
X_test.drop(['BedroomAbvGr'],axis=1,inplace=True)
X_valid.drop(['BedroomAbvGr'],axis=1,inplace=True)

#2ndFlrSF

features = ['2ndFlrSF','HouseStyle_2Story','MSSubClass_60']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['MSSubClass_60'],axis=1,inplace=True)
X_test.drop(['MSSubClass_60'],axis=1,inplace=True)
X_valid.drop(['MSSubClass_60'],axis=1,inplace=True)
X_train.drop(['HouseStyle_2Story'],axis=1,inplace=True)
X_test.drop(['HouseStyle_2Story'],axis=1,inplace=True)
X_valid.drop(['HouseStyle_2Story'],axis=1,inplace=True)

# Garage  Qual

features = ['GarageCond','GarageQual']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['GarageCond'],axis=1,inplace=True)
X_test.drop(['GarageCond'],axis=1,inplace=True)
X_valid.drop(['GarageCond'],axis=1,inplace=True)

# neigh c3

features = ['Neighborhood_CollgCr','NeighGroup_C3']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['Neighborhood_CollgCr'],axis=1,inplace=True)
X_test.drop(['Neighborhood_CollgCr'],axis=1,inplace=True)
X_valid.drop(['Neighborhood_CollgCr'],axis=1,inplace=True)

# gilbert
features = ['Neighborhood_Gilbert','NeighGroup_BNS']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['Neighborhood_Gilbert'],axis=1,inplace=True)
X_test.drop(['Neighborhood_Gilbert'],axis=1,inplace=True)
X_valid.drop(['Neighborhood_Gilbert'],axis=1,inplace=True)

# bsmt

features = ['BsmtFinType1_ALQ','BsmtFinSF1']
feat_list = list_trans(features)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['BsmtFinType1_ALQ'],axis=1,inplace=True)
X_test.drop(['BsmtFinType1_ALQ'],axis=1,inplace=True)
X_valid.drop(['BsmtFinType1_ALQ'],axis=1,inplace=True)

m = RandomForestRegressor(max_features = 0.5,n_estimators = 50 ,oob_score = True,min_samples_leaf=3)
m.fit(X_train,y_train)
print_score(m)

# test variables
feats = ['LotFrontage']

print_score(m)

for f in feats:
    for r in range (1,12,1):
        if r == 1:
            print(f)
            X_train_copy = X_train.copy()
            X_valid_copy = X_valid.copy()
            X_train_copy.drop(f,axis=1)
            X_valid_copy.drop(f,axis=1)
            m2 = RandomForestRegressor(max_features = 0.5,n_estimators = 150,min_samples_leaf=3, n_jobs =-1,oob_score = True)
            m2.fit(X_train_copy,y_train)
            print_score(m2)
        elif r != 1:
            X_train_copy = X_train.copy()
            X_valid_copy = X_valid.copy()
            X_train_copy.drop(f,axis=1)
            X_valid_copy.drop(f,axis=1)
            m2 = RandomForestRegressor(max_features = 0.5,n_estimators = 150,min_samples_leaf=3, n_jobs =-1,oob_score = True)
            m2.fit(X_train_copy,y_train)
            print_score(m2)
    
# drop as needed

X_train.drop(['CentralAir_Y'],axis=1,inplace=True)
X_test.drop(['CentralAir_Y'],axis=1,inplace=True)
X_valid.drop(['CentralAir_Y'],axis=1,inplace=True)

X_train.drop(['BsmtCond'],axis=1,inplace=True)
X_test.drop(['BsmtCond'],axis=1,inplace=True)
X_valid.drop(['BsmtCond'],axis=1,inplace=True)


X_train.drop(['1stFlrSF'],axis=1,inplace=True)
X_test.drop(['1stFlrSF'],axis=1,inplace=True)
X_valid.drop(['1stFlrSF'],axis=1,inplace=True)

X_train.drop(['BsmtExposure_Gd'],axis=1,inplace=True)
X_test.drop(['BsmtExposure_Gd'],axis=1,inplace=True)
X_valid.drop(['BsmtExposure_Gd'],axis=1,inplace=True)

X_train.drop(['LotFrontage'],axis=1,inplace=True)
X_test.drop(['LotFrontage'],axis=1,inplace=True)
X_valid.drop(['LotFrontage'],axis=1,inplace=True)

X_train.drop(['Electrical_SBrkr'],axis=1,inplace=True)
X_test.drop(['Electrical_SBrkr'],axis=1,inplace=True)
X_valid.drop(['Electrical_SBrkr'],axis=1,inplace=True)

X_train.drop(['GarageFinish_RFn'],axis=1,inplace=True)
X_test.drop(['GarageFinish_RFn'],axis=1,inplace=True)
X_valid.drop(['GarageFinish_RFn'],axis=1,inplace=True)
X_train.drop(['ExterQual'],axis=1,inplace=True)
X_test.drop(['ExterQual'],axis=1,inplace=True)
X_valid.drop(['ExterQual'],axis=1,inplace=True)
X_train.drop(['Exterior1st_VinylSd'],axis=1,inplace=True)
X_test.drop(['Exterior1st_VinylSd'],axis=1,inplace=True)
X_valid.drop(['Exterior1st_VinylSd'],axis=1,inplace=True)
X_train.drop(['MasVnrArea'],axis=1,inplace=True)
X_test.drop(['MasVnrArea'],axis=1,inplace=True)
X_valid.drop(['MasVnrArea'],axis=1,inplace=True)
X_train.drop(['Neighborhood_BrkSide'],axis=1,inplace=True)
X_test.drop(['Neighborhood_BrkSide'],axis=1,inplace=True)
X_valid.drop(['Neighborhood_BrkSide'],axis=1,inplace=True)
X_train.drop(['LotShape_Reg'],axis=1,inplace=True)
X_test.drop(['LotShape_Reg'],axis=1,inplace=True)
X_valid.drop(['LotShape_Reg'],axis=1,inplace=True)
X_train.drop(['YearRemodAdd'],axis=1,inplace=True)
X_test.drop(['YearRemodAdd'],axis=1,inplace=True)
X_valid.drop(['YearRemodAdd'],axis=1,inplace=True)

row = X_valid.values[None,0]; row
prediction,bias,contributions = ti.predict(m,row)
data = pd.DataFrame([o for o in zip(X_train.columns,X_valid.iloc[0],contributions[0])])
contributions[0].sum()
            
rmsle(X_train,y_train)
rmsle(X_valid,y_valid)

df_corr_mat = feature_corr_matrix(X_train)
df_corr_mat = df_corr_mat.dropna(axis='columns',how='all')
df_corr_mat = df_corr_mat.dropna()
df_corr_mat = df_corr_mat.values
corr_condensed = hc.distance.squareform(1-df_corr_mat)
z = hc.linkage(corr_condensed,method='average')
fig = plt.figure(figsize =(20,10))
dendrogram = hc.dendrogram(z,labels= X_train.columns,orientation = 'left',leaf_font_size =8)
plt.show()

plot_corr_heatmap(X_train)

# check to see if validation data set is overfitted or not

X_valid['Class'] = 1
X_train['Class'] = 0

new_df = pd.concat([X_valid,X_train])

m = RandomForestClassifier(n_estimators = 40, min_samples_leaf=3, max_features=0.5,n_jobs =-1, oob_score= True)

y_val = new_df['Class']
new_df.drop(['Class'],axis=1,inplace=True)
m.fit(new_df,y_val)
m.oob_score_

fi = rf_feat_importance(m,new_df); fi[:10]
feats = ['LotArea','BsmtUnfSF','GarageArea','FirstFlrSF','GarageYrBlt']
X_valid[feats].describe()
X_train[feats].describe()

X_valid.drop(['Class'],axis=1,inplace=True)
X_train.drop(['Class'],axis=1,inplace=True)

# submit predictions

current_wd = os.getcwd()
y_pred = m.predict(X_test)
df_test['SalePrice'] = y_pred
submissions = df_test[['Id','SalePrice']]
submissions['Id'] = submissions['Id'].astype(int)
submissions.to_csv(current_wd + '\\lr_submissions.csv',encoding='utf-8',index=False)










