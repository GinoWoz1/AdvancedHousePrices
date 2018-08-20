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


warnings.filterwarnings('ignore')
%matplotlib inline

current_loc = os.getcwd()

if 'jjonus' in current_loc:
    current_wd = ('C:\\Users\\jjonus\\Google Drive\\Kaggle\\Advanced House Prices')
elif 'jstnj' in current_loc:
    current_wd = ('C:\\Users\\jstnj\\Google Drive\\Kaggle\\Advanced House Prices')
    
df_train = pd.read_csv(current_wd +'\\train.csv')   
df_test = pd.read_csv(current_wd +'\\test.csv')

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

# evaluation metrics and model vetting function

def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train),y_train),rmse(m.predict(X_valid),y_valid),m.score(X_train,y_train),m.score(X_valid,y_valid)]
    if hasattr(m,'oob_score_'): res.append(m.oob_score_)
    print(res)

def rmsle(X, y): 
    y_pred = m.predict(X)
    assert len(y) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y))**2))

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

X_test = test.copy()
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_valid = pd.get_dummies(X_valid)

sns.distplot(y_train)
sns.distplot(y_valid)
fig = plt.figure()

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

m = RandomForestRegressor(max_features = 0.5,n_estimators = 150 ,oob_score = True,min_samples_leaf=3)
m.fit(X_train,y_train)
print(m.oob_score_)

# examine tree
draw_tree(m.estimators_[0],X_train,precision=3)

# mutual information data

df_mi1 = pd.DataFrame.from_dict(new_dict,orient = 'index')
df_mi2 = pd.DataFrame.from_dict(new_dict2,orient= 'index')

df_mi1.to_csv(current_wd + '\\mi1.csv')
df_mi2.to_csv(current_wd + '\\mi2.csv')

 
X_train = X_train[['OverallQual'	,'TotalBsmtSF'	,'GarageCars'	,'Baths'	,'GrLivArea'	,'ExterQual_TA'	,'YearBuilt'	,'ExterQual_Gd'	,'1stFlrSF'	,'YearRemodAdd'	,'FullBath'	,'KitchenQual_TA'	,'Fireplaces'	,'BsmtQual_TA'	,'BsmtQual_Gd'	,'KitchenQual_Gd'	,'BsmtUnfSF'	,'Foundation_PConc'	,'LotFrontage'	,'HeatingQC_Ex'	,'GarageType_Attchd'	,'MSSubClass_60'	,'BsmtQual_Ex'	,'TotRmsAbvGrd'	,'GarageFinish_Unf'	,'2ndFlrSF'	,'BsmtFinType1_GLQ'	,'GarageType_Detchd'	,'OpenPorchSF'	,'GarageFinish_Fin'	,'HeatingQC_TA'	,'BsmtFinSF1'	,'LotArea'	,'GarageQual_TA'	,'Exterior1st_VinylSd'	,'Neighborhood_NridgHt'	,'FireplaceQu_TA'	,'FireplaceQu_Gd'	,'Foundation_CBlock'	,'OverallCond'	,'MasVnrArea'	,'MSZoning_RM'	,'HouseStyle_2Story'	,'Exterior2nd_VinylSd'	,'MasVnrType_None'	,'GarageCond_TA'	,'MSSubClass_30'	,'BedroomAbvGr'	,'GarageCond_None'	,'CentralAir_Y'	,'GarageFinish_RFn'	,'GarageQual_None'	,'KitchenQual_Ex'	,'GarageFinish_None'	,'GarageType_None'	,'Neighborhood_CollgCr'	,'Neighborhood_NWAmes'	,'LotShape_IR1'	,'LotShape_Reg'	,'Neighborhood_Gilbert'	,'MasVnrType_Stone'	,'Neighborhood_NoRidge'	,'Neighborhood_NAmes'	,'Foundation_BrkTil'	,'PavedDrive_Y'	,'KitchenQual_Fa'	,'BsmtFinType1_ALQ'	,'Condition1_Feedr'	,'BsmtFinType1_None'	,'BsmtExposure_None'	,'Exterior2nd_Wd Sdng'	,'BsmtFinType2_None'	,'MSZoning_RL'	,'BsmtCond_None'	,'BsmtQual_None'	,'MSSubClass_90'	,'BldgType_Duplex'	,'Electrical_FuseA'	,'EnclosedPorch'	,'Fence_None'	,'ExterQual_Ex'	,'KitchenAbvGr'	,'Neighborhood_Somerst'	,'MSSubClass_160'	,'Electrical_SBrkr']]
X_valid = X_valid[['OverallQual'	,'TotalBsmtSF'	,'GarageCars'	,'Baths'	,'GrLivArea'	,'ExterQual_TA'	,'YearBuilt'	,'ExterQual_Gd'	,'1stFlrSF'	,'YearRemodAdd'	,'FullBath'	,'KitchenQual_TA'	,'Fireplaces'	,'BsmtQual_TA'	,'BsmtQual_Gd'	,'KitchenQual_Gd'	,'BsmtUnfSF'	,'Foundation_PConc'	,'LotFrontage'	,'HeatingQC_Ex'	,'GarageType_Attchd'	,'MSSubClass_60'	,'BsmtQual_Ex'	,'TotRmsAbvGrd'	,'GarageFinish_Unf'	,'2ndFlrSF'	,'BsmtFinType1_GLQ'	,'GarageType_Detchd'	,'OpenPorchSF'	,'GarageFinish_Fin'	,'HeatingQC_TA'	,'BsmtFinSF1'	,'LotArea'	,'GarageQual_TA'	,'Exterior1st_VinylSd'	,'Neighborhood_NridgHt'	,'FireplaceQu_TA'	,'FireplaceQu_Gd'	,'Foundation_CBlock'	,'OverallCond'	,'MasVnrArea'	,'MSZoning_RM'	,'HouseStyle_2Story'	,'Exterior2nd_VinylSd'	,'MasVnrType_None'	,'GarageCond_TA'	,'MSSubClass_30'	,'BedroomAbvGr'	,'GarageCond_None'	,'CentralAir_Y'	,'GarageFinish_RFn'	,'GarageQual_None'	,'KitchenQual_Ex'	,'GarageFinish_None'	,'GarageType_None'	,'Neighborhood_CollgCr'	,'Neighborhood_NWAmes'	,'LotShape_IR1'	,'LotShape_Reg'	,'Neighborhood_Gilbert'	,'MasVnrType_Stone'	,'Neighborhood_NoRidge'	,'Neighborhood_NAmes'	,'Foundation_BrkTil'	,'PavedDrive_Y'	,'KitchenQual_Fa'	,'BsmtFinType1_ALQ'	,'Condition1_Feedr'	,'BsmtFinType1_None'	,'BsmtExposure_None'	,'Exterior2nd_Wd Sdng'	,'BsmtFinType2_None'	,'MSZoning_RL'	,'BsmtCond_None'	,'BsmtQual_None'	,'MSSubClass_90'	,'BldgType_Duplex'	,'Electrical_FuseA'	,'EnclosedPorch'	,'Fence_None'	,'ExterQual_Ex'	,'KitchenAbvGr'	,'Neighborhood_Somerst'	,'MSSubClass_160'	,'Electrical_SBrkr']]
X_test = X_test[['OverallQual'	,'TotalBsmtSF'	,'GarageCars'	,'Baths'	,'GrLivArea'	,'ExterQual_TA'	,'YearBuilt'	,'ExterQual_Gd'	,'1stFlrSF'	,'YearRemodAdd'	,'FullBath'	,'KitchenQual_TA'	,'Fireplaces'	,'BsmtQual_TA'	,'BsmtQual_Gd'	,'KitchenQual_Gd'	,'BsmtUnfSF'	,'Foundation_PConc'	,'LotFrontage'	,'HeatingQC_Ex'	,'GarageType_Attchd'	,'MSSubClass_60'	,'BsmtQual_Ex'	,'TotRmsAbvGrd'	,'GarageFinish_Unf'	,'2ndFlrSF'	,'BsmtFinType1_GLQ'	,'GarageType_Detchd'	,'OpenPorchSF'	,'GarageFinish_Fin'	,'HeatingQC_TA'	,'BsmtFinSF1'	,'LotArea'	,'GarageQual_TA'	,'Exterior1st_VinylSd'	,'Neighborhood_NridgHt'	,'FireplaceQu_TA'	,'FireplaceQu_Gd'	,'Foundation_CBlock'	,'OverallCond'	,'MasVnrArea'	,'MSZoning_RM'	,'HouseStyle_2Story'	,'Exterior2nd_VinylSd'	,'MasVnrType_None'	,'GarageCond_TA'	,'MSSubClass_30'	,'BedroomAbvGr'	,'GarageCond_None'	,'CentralAir_Y'	,'GarageFinish_RFn'	,'GarageQual_None'	,'KitchenQual_Ex'	,'GarageFinish_None'	,'GarageType_None'	,'Neighborhood_CollgCr'	,'Neighborhood_NWAmes'	,'LotShape_IR1'	,'LotShape_Reg'	,'Neighborhood_Gilbert'	,'MasVnrType_Stone'	,'Neighborhood_NoRidge'	,'Neighborhood_NAmes'	,'Foundation_BrkTil'	,'PavedDrive_Y'	,'KitchenQual_Fa'	,'BsmtFinType1_ALQ'	,'Condition1_Feedr'	,'BsmtFinType1_None'	,'BsmtExposure_None'	,'Exterior2nd_Wd Sdng'	,'BsmtFinType2_None'	,'MSZoning_RL'	,'BsmtCond_None'	,'BsmtQual_None'	,'MSSubClass_90'	,'BldgType_Duplex'	,'Electrical_FuseA'	,'EnclosedPorch'	,'Fence_None'	,'ExterQual_Ex'	,'KitchenAbvGr'	,'Neighborhood_Somerst'	,'MSSubClass_160'	,'Electrical_SBrkr']]

m = RandomForestRegressor(max_features = 0.5,n_estimators = 100 ,oob_score = True,min_samples_leaf=3)
m.fit(X_train,y_train)
print_score(m)

train_sizes,train_scores,test_scores = learning_curve(estimator=m,X=X_train,y=y_train,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=-1)

train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)

plt.plot(train_sizes,train_mean,color = 'blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(train_sizes,train_mean + train_std,train_mean - train_std,alpha=0.15,color='blue')
plt.plot(train_sizes,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='validation accuracy')
plt.fill_between(train_sizes,test_mean + test_std,test_mean - test_std,alpha=0.15,color='blue')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8,1.1])
plt.show()

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
matrix = feature_corr_matrix(X_train)

# test dropcol_importance for grouped importances ( to ascertain individual drop-ability)

df = dropcol_importances(m,X_valid,y_valid)

# Investigate correlation on no basement, duplex and MSSubClass 90

features = [['BsmtCond_None', 'BsmtQual_None', 'BsmtFinType2_None','BsmtExposure_None','BsmtFinType1_None'],  
            ['BsmtQual_None'], 
            ['BldgType_Duplex','MSSubClass_90','KitchenAbvGr'],['MSSubClass_90','KitchenAbvGr'],'BldgType_Duplex','MSSubClass_90','KitchenAbvGr']

I = importances(m, X_valid, y_valid,features = features)

X_train.drop(['BsmtCond_None','BsmtQual_None', 'BsmtFinType2_None','BsmtExposure_None','BsmtFinType1_None','MSSubClass_90'],axis=1,inplace=True)
X_test.drop(['BsmtCond_None', 'BsmtQual_None','BsmtFinType2_None','BsmtExposure_None','BsmtFinType1_None','MSSubClass_90'],axis=1,inplace=True)
X_valid.drop(['BsmtCond_None', 'BsmtQual_None','BsmtFinType2_None','BsmtExposure_None','BsmtFinType1_None','MSSubClass_90'],axis=1,inplace=True)

m = RandomForestRegressor(max_features = 0.5,n_estimators = 150 ,oob_score = True,min_samples_leaf=3)
m.fit(X_train,y_train)
print_score(m)

features = [['GarageType_None','GarageFinish_None','GarageCond_None'],  
            ['HouseStyle_2Story','MSSubClass_60'],['Exterior1st_VinylSd','Exterior2nd_VinylSd'],
            ['BldgType_Duplex','KitchenAbvGr'],'GarageType_None','GarageFinish_None','GarageCond_None','HouseStyle_2Story','MSSubClass_60','Exterior1st_VinylSd','Exterior2nd_VinylSd']

df = dropcol_importances(m,X_valid,y_valid)
I = importances(m, X_valid, y_valid,features = features)

X_train.drop(['GarageType_None','GarageFinish_None','GarageCond_None','Exterior2nd_VinylSd'],axis=1,inplace=True)
X_test.drop(['GarageType_None','GarageFinish_None','GarageCond_None','Exterior2nd_VinylSd'],axis=1,inplace=True)
X_valid.drop(['GarageType_None','GarageFinish_None','GarageCond_None','Exterior2nd_VinylSd'],axis=1,inplace=True)

m = RandomForestRegressor(max_features = 0.5,n_estimators = 150 ,oob_score = True,min_samples_leaf=3)
m.fit(X_train,y_train)
print_score(m)

# examine next set of collinear variables

df = dropcol_importances(m,X_valid,y_valid)
I = importances(m, X_valid, y_valid,features = features)

features = ['Baths','TotalBsmtSF','GarageCond_TA','HouseStyle_2Story','TotRmsAbvGrd','GrLivArea']
features = ['TotalSF','GrLivArea','TotalBsmtSF']

feat_list = list_trans(features)

I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['Baths','GarageCond_TA','HouseStyle_2Story'],axis=1,inplace=True)
X_test.drop(['Baths','GarageCond_TA','HouseStyle_2Story'],axis=1,inplace=True)
X_valid.drop(['Baths','GarageCond_TA','HouseStyle_2Story'],axis=1,inplace=True)

X_train['TotalSF'] = X_train['GrLivArea'] + X_train['TotalBsmtSF']
X_valid['TotalSF'] = X_valid['GrLivArea'] + X_valid['TotalBsmtSF']
X_test['TotalSF'] = X_test['GrLivArea'] + X_test['TotalBsmtSF']
X_train.drop(['GrLivArea','TotalBsmtSF'],axis=1,inplace=True)
X_valid.drop(['GrLivArea','TotalBsmtSF'],axis=1,inplace=True)
X_test.drop(['GrLivArea','TotalBsmtSF'],axis=1,inplace=True)

m = RandomForestRegressor(max_features = 0.5,n_estimators = 150 ,oob_score = True,min_samples_leaf=3)
m.fit(X_train,y_train)
print_score(m)

rmsle(X_train,y_train)
rmsle(X_valid,y_valid)

draw_tree(m.estimators_[0],X_train,precision=3)

df_corr_mat = feature_corr_matrix(X_train)
df_corr_mat = df_corr_mat.dropna(axis='columns',how='all')
df_corr_mat = df_corr_mat.dropna()
df_corr_mat = df_corr_mat.values
corr_condensed = hc.distance.squareform(1-df_corr_mat)
z = hc.linkage(corr_condensed,method='average')
fig = plt.figure(figsize =(20,10))
dendrogram = hc.dendrogram(z,labels= X_train.columns,orientation = 'left',leaf_font_size =8)
plt.show()

# examime correlation matrix manually

matrix = feature_corr_matrix(X_train)

features = [['BsmtQual_Ex','ExterQual_Ex','KitchenQual_Ex'],['BsmtQual_Ex','ExterQual_Ex'],['KitchenQual_Ex','ExterQual_Ex'],'BsmtQual_Ex','ExterQual_Ex','KitchenQual_Ex']

df = dropcol_importances(m,X_valid,y_valid)
I = importances(m, X_valid, y_valid,features = features)

features = [['BsmtQual_Ex','ExterQual_Ex','KitchenQual_Ex','HeatingQC_Ex'],['BsmtQual_Ex','ExterQual_Ex','HeatingQC_Ex'],['KitchenQual_Ex','ExterQual_Ex'],'HeatingQC_Ex','BsmtQual_Ex','ExterQual_Ex','KitchenQual_Ex','ExcellentQuality']

df = dropcol_importances(m,X_valid,y_valid)
I = importances(m, X_valid, y_valid,features = list_o_lists)

X_train.drop(['BsmtQual_Ex','HeatingQC_Ex'],axis=1,inplace=True)
X_test.drop(['BsmtQual_Ex','HeatingQC_Ex'],axis=1,inplace=True)
X_valid.drop(['BsmtQual_Ex','HeatingQC_Ex'],axis=1,inplace=True)

m = RandomForestRegressor(max_features = 0.5,n_estimators = 150 ,oob_score = True,min_samples_leaf=3)
m.fit(X_train,y_train)
print_score(m)

# check next set of variables
df = dropcol_importances(m,X_valid,y_valid)

df_corr_mat = feature_corr_matrix(X_train)
df_corr_mat = df_corr_mat.dropna(axis='columns',how='all')
df_corr_mat = df_corr_mat.dropna()
df_corr_mat = df_corr_mat.values
corr_condensed = hc.distance.squareform(1-df_corr_mat)
z = hc.linkage(corr_condensed,method='average')
fig = plt.figure(figsize =(20,10))
dendrogram = hc.dendrogram(z,labels= X_train.columns,orientation = 'left',leaf_font_size =8)
plt.show()

# examine These 8 highly correlated variables

matrix = feature_corr_matrix(X_train)

features = [['Foundation_PConc','YearBuilt','YearRemodAdd','Exterior1st_VinylSd','KitchenQual_Gd','ExterQual_Gd','BsmtQual_Gd']]

feat_list = list_trans(features)

df = dropcol_importances(m,X_valid,y_valid)
I = importances(m, X_valid, y_valid,features = new_list)

X_train.drop(['BsmtQual_Gd'],axis=1,inplace=True)
X_test.drop(['BsmtQual_Gd'],axis=1,inplace=True)
X_valid.drop(['BsmtQual_Gd'],axis=1,inplace=True)

m = RandomForestRegressor(max_features = 0.5,n_estimators = 150 ,oob_score = True,min_samples_leaf=3)
m.fit(X_train,y_train)
print_score(m)

rmsle(X_train,y_train)
rmsle(X_valid,y_valid)

# check TA qualities

features = ['ExterQual_TA','BsmtQual_TA','HeatingQC_TA','KitchenQual_TA']

feat_list = list_trans(features)

df = dropcol_importances(m,X_valid,y_valid)
I = importances(m, X_valid, y_valid,features = feat_list)

X_train.drop(['BsmtQual_TA', 'HeatingQC_TA'],axis=1,inplace=True)
X_test.drop(['BsmtQual_TA', 'HeatingQC_TA'],axis=1,inplace=True)
X_valid.drop(['BsmtQual_TA', 'HeatingQC_TA'],axis=1,inplace=True)


m = RandomForestRegressor(max_features = 0.5,n_estimators = 150 ,oob_score = True,min_samples_leaf=3)
m.fit(X_train,y_train)
print_score(m)

rmsle(X_train,y_train)
rmsle(X_valid,y_valid)

df = dropcol_importances(m,X_valid,y_valid)

X_train.drop(['1stFlrSF'],axis=1,inplace=True)
X_test.drop(['1stFlrSF'],axis=1,inplace=True)
X_valid.drop(['1stFlrSF'],axis=1,inplace=True)

df = dropcol_importances(m,X_valid,y_valid)

X_train.drop(['MasVnrArea'],axis=1,inplace=True)
X_test.drop(['MasVnrArea'],axis=1,inplace=True)
X_valid.drop(['MasVnrArea'],axis=1,inplace=True)

feats = ['LotShape_IR1','Foundation_BrkTil','LotFrontage']

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
            
# start dropping in sequence (each run) the highest value oob increase I get from dropping
            
X_train.drop(['1stFlrSF'],axis=1,inplace=True)
X_test.drop(['1stFlrSF'],axis=1,inplace=True)
X_valid.drop(['1stFlrSF'],axis=1,inplace=True)
X_train.drop(['MasVnrArea'],axis=1,inplace=True)
X_test.drop(['MasVnrArea'],axis=1,inplace=True)
X_valid.drop(['MasVnrArea'],axis=1,inplace=True)
X_train.drop(['2ndFlrSF'],axis=1,inplace=True)
X_test.drop(['2ndFlrSF'],axis=1,inplace=True)
X_valid.drop(['2ndFlrSF'],axis=1,inplace=True)
X_train.drop(['GarageType_Attchd'],axis=1,inplace=True)
X_test.drop(['GarageType_Attchd'],axis=1,inplace=True)
X_valid.drop(['GarageType_Attchd'],axis=1,inplace=True)
X_train.drop(['GarageFinish_Unf'],axis=1,inplace=True)
X_test.drop(['GarageFinish_Unf'],axis=1,inplace=True)
X_valid.drop(['GarageFinish_Unf'],axis=1,inplace=True)
X_train.drop(['TotRmsAbvGrd'],axis=1,inplace=True)
X_test.drop(['TotRmsAbvGrd'],axis=1,inplace=True)
X_valid.drop(['TotRmsAbvGrd'],axis=1,inplace=True)
X_train.drop(['LotShape_Reg'],axis=1,inplace=True)
X_test.drop(['LotShape_Reg'],axis=1,inplace=True)
X_valid.drop(['LotShape_Reg'],axis=1,inplace=True)
X_train.drop(['KitchenQual_TA','HeatingQC_TA'],axis=1,inplace=True)
X_test.drop(['KitchenQual_TA','HeatingQC_TA'],axis=1,inplace=True)
X_valid.drop(['KitchenQual_TA','HeatingQC_TA'],axis=1,inplace=True)
X_train.drop(['BsmtFinType1_ALQ'],axis=1,inplace=True)
X_test.drop(['BsmtFinType1_ALQ'],axis=1,inplace=True)
X_valid.drop(['BsmtFinType1_ALQ'],axis=1,inplace=True)
X_train.drop(['BsmtFinSF1'],axis=1,inplace=True)
X_test.drop(['BsmtFinSF1'],axis=1,inplace=True)
X_valid.drop(['BsmtFinSF1'],axis=1,inplace=True)
X_train.drop(['Electrical_SBrkr'],axis=1,inplace=True)
X_test.drop(['Electrical_SBrkr'],axis=1,inplace=True)
X_valid.drop(['Electrical_SBrkr'],axis=1,inplace=True)
X_train.drop(['BsmtFinType1_GLQ'],axis=1,inplace=True)
X_test.drop(['BsmtFinType1_GLQ'],axis=1,inplace=True)
X_valid.drop(['BsmtFinType1_GLQ'],axis=1,inplace=True)
X_train.drop(['GarageQual_None'],axis=1,inplace=True)
X_test.drop(['GarageQual_None'],axis=1,inplace=True)
X_valid.drop(['GarageQual_None'],axis=1,inplace=True)
X_train.drop(['MasVnrType_None'],axis=1,inplace=True)
X_test.drop(['MasVnrType_None'],axis=1,inplace=True)
X_valid.drop(['MasVnrType_None'],axis=1,inplace=True)
X_train.drop(['Condition1_Feedr'],axis=1,inplace=True)
X_test.drop(['Condition1_Feedr'],axis=1,inplace=True)
X_valid.drop(['Condition1_Feedr'],axis=1,inplace=True)
X_train.drop(['FireplaceQu_Gd','Electrical_FuseA'],axis=1,inplace=True)
X_test.drop(['FireplaceQu_Gd','Electrical_FuseA'],axis=1,inplace=True)
X_valid.drop(['FireplaceQu_Gd','Electrical_FuseA'],axis=1,inplace=True)
X_train.drop(['FireplaceQu_TA'],axis=1,inplace=True)
X_test.drop(['FireplaceQu_TA'],axis=1,inplace=True)
X_valid.drop(['FireplaceQu_TA'],axis=1,inplace=True)

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

# test removing 0 mutual information columns

param_range = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]

param_grid = [{'n_estimators': [50,60,70,80,90,100,110,120,130,140,160,170,180,190,200],
               'max_depth':[3,5,7,9,11,13],
                 'max_features': [0.3,0.4,0.5],
                 'min_samples_leaf':[3,5]}]

gs = GridSearchCV(estimator = m,param_grid=param_grid,
                  scoring = 'neg_mean_squared_error',
                  cv=10,n_jobs=-1)

gs = gs.fit(X_train,y_train)
gs.best_score_
gs.best_params_

m = RandomForestRegressor(max_features = 0.5, n_estimators = 150 ,oob_score = True,min_samples_leaf=3)
m.fit(X_train,y_train)
print_score(m)


draw_tree(m.estimators_[0],X_train,precision=3)

# plot predict vs actual

import matplotlib.pyplot as plt

y_pred_train = m.predict(X_train)
plt.scatter(y_pred_train,y_train)# check how random validation set is

y_pred_valid = m.predict(X_valid)
plt.scatter(y_pred_valid,y_valid)# check how random validation set is

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

# analyze importance on validation set 


y_pred = m.predict(X_test)

df_test['SalePrice'] = y_pred
submissions = df_test[['Id','SalePrice']]
submissions['Id'] = submissions['Id'].astype(int)
submissions.to_csv(current_wd + '\\lr_submissions.csv',encoding='utf-8',index=False)










