# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 10:58:19 2018

@author: jjonus
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from scipy import stats
import warnings
import missingno as msno
from fancyimpute import MICE
from scipy import stats
from scipy.stats import norm, skew
from sklearn.linear_model import Lasso,Ridge,ElasticNet
from math import ceil
from fastai.imports import *  # fast ai library
from pandas_summary import * 

# start modelling - check the learning and validation curve along the way  
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.svm import LinearSVR,SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve,validation_curve,cross_val_score,GridSearchCV,train_test_split,RepeatedKFold
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from statsmodels.tools.tools import add_constant

warnings.filterwarnings('ignore')
%matplotlib inline

# function to find outliers

def find_outliers(model, X, y, sigma=3):

    # predict y values using model
    try:
        y_pred = pd.Series(model.predict(X), index=y.index)
    # if predicting fails, try fitting the model first
    except:
        model.fit(X,y)
        y_pred = pd.Series(model.predict(X), index=y.index)
        
    # calculate residuals between the model prediction and true y values
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # calculate z statistic, define outliers to be where |z|>sigma
    z = (resid - mean_resid)/std_resid    
    outliers = z[abs(z)>sigma].index
    
    # print and plot the results
    print('R2=',model.score(X,y))
    print('rmse=',rmse(y, y_pred))
    print('---------------------------------------')

    print('mean of residuals:',mean_resid)
    print('std of residuals:',std_resid)
    print('---------------------------------------')

    print(len(outliers),'outliers:')
    print(outliers.tolist())

    plt.figure(figsize=(15,5))
    ax_131 = plt.subplot(1,3,1)
    plt.plot(y,y_pred,'.')
    plt.plot(y.loc[outliers],y_pred.loc[outliers],'ro')
    plt.legend(['Accepted','Outlier'])
    plt.xlabel('y')
    plt.ylabel('y_pred');

    ax_132=plt.subplot(1,3,2)
    plt.plot(y,y-y_pred,'.')
    plt.plot(y.loc[outliers],y.loc[outliers]-y_pred.loc[outliers],'ro')
    plt.legend(['Accepted','Outlier'])
    plt.xlabel('y')
    plt.ylabel('y - y_pred');

    ax_133=plt.subplot(1,3,3)
    z.plot.hist(bins=50,ax=ax_133)
    z.loc[outliers].plot.hist(color='r',bins=50,ax=ax_133)
    plt.legend(['Accepted','Outlier'])
    plt.xlabel('z')
    
    plt.savefig('outliers.png')
    
    return outliers

# function for one-way anova ( to find important variables)

def anova(group,value):
    # select columns of interest, and remove any rows with nan values
    data = train[[group,value]]
    data = data[~(data[group].isnull() | data[value].isnull())]
    
    # stats across all data
    tot_groups = data[group].nunique() # no. of groups
    len_data = len(data) # total sample size of houses (all groups)
    mean_data = data[value].mean() # mean across all groups
    df_betwn = tot_groups - 1 # degrees of freedom betwn grps
    df_within = len_data - tot_groups # degrees of freedom within grps
    
    # per group stats
    n_in_group = data.groupby(group)[value].count() # no. houses in group
    mean_group = data.groupby(group)[value].mean() # mean value in this group
    
    # between-group variability
    betwn_var = n_in_group*((mean_group - mean_data)**2)
    betwn_var = float(betwn_var.sum())/df_betwn
    
    # within-group variability
    within_var = 0
    for grp in data[group].unique():
        samples = data.loc[data[group]==grp, value]
        within_var += ((samples-mean_group[grp])**2).sum()
        
    within_var = float(within_var)/df_within
    
    #F-test statistic
    F = betwn_var/within_var
    
    # p-value
    p = stats.f.sf(F, df_betwn, df_within)
    
    return p   

def train_model(model, param_grid=[], X=[], y=[], 
                splits=5, repeats=5):

    # get unmodified training data, unless data to use already specified
    if len(y)==0:
        X,y = get_training_data()
    
    # create cross-validation method
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
    
    # perform a grid search if param_grid given
    if len(param_grid)>0:
        # setup grid search parameters
        gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                               scoring=rmse_scorer,
                               verbose=1, return_train_score=True)

        # search the grid
        gsearch.fit(X,y)

        # extract best model from the grid
        model = gsearch.best_estimator_        
        best_idx = gsearch.best_index_

        # get cv-scores for best model
        grid_results = pd.DataFrame(gsearch.cv_results_)       
        cv_mean = abs(grid_results.loc[best_idx,'mean_test_score'])
        cv_std = grid_results.loc[best_idx,'std_test_score']

    # no grid search, just cross-val score for given model    
    else:
        grid_results = []
        cv_results = cross_val_score(model, X, y, scoring=rmse_scorer, cv=rkfold)
        cv_mean = abs(np.mean(cv_results))
        cv_std = np.std(cv_results)
    
    # combine mean and std cv-score in to a pandas series
    cv_score = pd.Series({'mean':cv_mean,'std':cv_std})

    # predict y using the fitted model
    y_pred = model.predict(X)
    
    # print stats on model performance         
    print('----------------------')
    print(model)
    print('----------------------')
    print('score=',model.score(X,y))
    print('rmse=',rmse(y, y_pred))
    print('cross_val: mean=',cv_mean,', std=',cv_std)
    
    # residual plots
    y_pred = pd.Series(y_pred,index=y.index)
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()
    z = (resid - mean_resid)/std_resid    
    n_outliers = sum(abs(z)>3)
    
    plt.figure(figsize=(15,5))
    ax_131 = plt.subplot(1,3,1)
    plt.plot(y,y_pred,'.')
    plt.xlabel('y')
    plt.ylabel('y_pred');
    plt.title('corr = {:.3f}'.format(np.corrcoef(y,y_pred)[0][1]))
    ax_132=plt.subplot(1,3,2)
    plt.plot(y,y-y_pred,'.')
    plt.xlabel('y')
    plt.ylabel('y - y_pred');
    plt.title('std resid = {:.3f}'.format(std_resid))
    
    ax_133=plt.subplot(1,3,3)
    z.plot.hist(bins=50,ax=ax_133)
    plt.xlabel('z')
    plt.title('{:.0f} samples with z>3'.format(n_outliers))

    return model, cv_score, grid_results

def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(x_tr),y_train),rmse(m.predict(x_val),y_valid),m.score(x_tr,y_train),m.score(x_val,y_valid)]
    print(res)
    
# read data in
   
url = 'https://github.com/GinoWoz1/AdvancedHousePrices/raw/master/'
    
df_train = pd.read_csv(url + 'train.csv')
df_test = pd.read_csv(url +'test.csv')
    
df_train.columns.to_series().groupby(df_train.dtypes).groups

# combine data sets first and clean data
ntrain = df_train.shape[0]
ntest = df_test.shape[0]
y_train = pd.Series(np.log(df_train.SalePrice.values))
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

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

# feature engineering on quality types
all_data['TotalSF'] = all_data['GrLivArea'] + all_data['TotalBsmtSF']

# change from number to categorical variable (set in string)
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

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

# Exter and KitchenQual Combined

all_data['KitchenExterQual'] = all_data['KitchenQual'] + all_data['ExterQual']

# boxplot lot frontage by neighborhood
plt.figure(figsize=(20,15))
sns.boxplot(x = "Neighborhood",y="LotFrontage",data = all_data)
plt.show()

# analyze which neighborhoods have null data
all_data[pd.isna(all_data['LotFrontage'])].groupby("Neighborhood").size().sort_values(ascending = False)

# datatypes
cols = all_data.loc[:,all_data.dtypes == np.int64].columns.tolist() 
all_data[cols] = all_data[cols].astype(float)
all_data.columns.to_series().groupby(all_data.dtypes).groups

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

# split back into train and test
train = all_data[:ntrain]
test = all_data[ntrain:]

# run imputation separately on each data set
numeric_data_train = train.select_dtypes(include = ['float64','int64'])
solver=MICE()
Imputed_dataframe_train= pd.DataFrame(data = solver.complete(numeric_data_train),columns = numeric_data_train.columns,index = numeric_data_train.index)
numeric_data_test = test.select_dtypes(include = ['float64','int64'])
solver=MICE()
Imputed_dataframe_test= pd.DataFrame(data = solver.complete(numeric_data_test),columns = numeric_data_test.columns,index = numeric_data_test.index)

# set back to main dataframe dataset
train['LotFrontage'] = Imputed_dataframe_train['LotFrontage']
test['LotFrontage'] = Imputed_dataframe_test['LotFrontage']

# normalize data
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats}).reset_index()

skewness.rename(columns = {'index':'feature'},inplace=True)

# remove values that shouldnt be transformed
skewness = skewness[skewness.feature != 'BsmtHalfBath']
skewness = skewness[skewness.feature != 'KitchenAbvGr']
skewness = skewness[skewness.feature != 'HalfBath']
skewness = skewness[skewness.feature != 'YearRemodAdd']
skewness = skewness[skewness.feature != 'OverallQual']
skewness = skewness[skewness.feature != 'YearBuilt']
skewness = skewness[skewness.feature != 'Id']
skewness = skewness[skewness.feature != 'GarageYrBlt']
skewness = skewness[skewness.feature != 'FullBath']
skewness = skewness[skewness.feature != 'GarageCars']
skewness = skewness[skewness.feature != 'Fireplaces']
skewness = skewness[skewness.feature != 'MoSold']
skewness = skewness[skewness.feature != 'YrSold']
skewness = skewness[skewness.feature != 'OverallCond']
skewness = skewness[skewness.feature != 'BedroomAbvGr']
skewness = skewness[skewness.feature != 'BsmtFullBath']
skewness = skewness[skewness.feature != 'HasVnrArea']
skewness = skewness[skewness.feature != 'IsDuplex']
skewness = skewness[skewness.feature != 'HasMiscVal']
skewness = skewness[skewness.feature != 'HasPconc']
skewness = skewness[skewness.feature != 'HasScreenPorch']
skewness = skewness[skewness.feature != 'HasTotalBsmtSF']
skewness = skewness[skewness.feature != 'HasVnrStone']
skewness = skewness[skewness.feature != 'HasWoodDeckSF']
skewness = skewness[skewness.feature != 'Has2ndFlrSF']
skewness = skewness[skewness.feature != 'HeatingEx']
skewness = skewness[skewness.feature != 'HasGarageArea']
skewness = skewness[skewness.feature != 'HasEnclosedPorch']
skewness = skewness[skewness.feature != 'HasFireplaces']
skewness = skewness[skewness.feature != 'HasMasVnrArea']
skewness = skewness[skewness.feature != 'HasOpenPorchSF']
skewness = skewness[skewness.feature != 'After91']

scaler = StandardScaler()
scale_features = skewness.feature.values.tolist()

train[scale_features] = scaler.fit_transform(train[scale_features])
test[scale_features] = scaler.fit_transform(test[scale_features])

# create feature for 1st or 2nd floor
train['HalfBathTransformed'] = train['HalfBath'] * 0.5
train['HalfBathBsmtTransformed'] = train['BsmtHalfBath'] * 0.5
train['Baths'] = train['FullBath'] + train['HalfBathTransformed'] + train['HalfBathBsmtTransformed']  + train['BsmtFullBath']


test['HalfBathTransformed'] = test['HalfBath'] * 0.5
test['HalfBathBsmtTransformed'] = test['BsmtHalfBath'] * 0.5
test['Baths'] = test['FullBath'] + test['HalfBathTransformed'] + test['HalfBathBsmtTransformed']  + test['BsmtFullBath']


# look at distribution of sale price
df_train['SalePrice'].describe()

# check histogram on sales price

sns.distplot(df_train['SalePrice'])

# check norm
sns.distplot(np.log(df_train['SalePrice']), fit=norm);
fig = plt.figure()
res = stats.probplot(np.log(df_train['SalePrice']), plot=plt)

# perform eda
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

# identify columns and their relation to sale price   

# grab category variables
dtypes = train.dtypes
cat_feats = train.dtypes[train.dtypes == "object"].index
numeric_feats = train.dtypes[train.dtypes != "object"].index

col_nunique = dict()

for col in numeric_feats:
    col_nunique[col] = train[col].nunique()
 
col_nunique = pd.Series(col_nunique)

cols_discrete = col_nunique[col_nunique<13].index.tolist()
cols_continuous = col_nunique[col_nunique>=13].index.tolist()

# plot categorical variables
fcols = 3
frows = ceil(len(cat_feats)/fcols)
plt.figure(figsize = (20,frows*4))

for i,col in enumerate(cat_feats):
    plt.subplot(frows,fcols,i+1)
    sns.violinplot(train[col],train['SalePrice'],innter = "stick")

p_col = dict()

for col in cat_feats:
    p_col[col] = anova(col,'SalePrice')

pd.Series(p_col).sort_values()

#Neighbourhood
plt.figure(figsize=(25,5))
sns.violinplot(x='Neighborhood',y='SalePrice',data=train)
plt.xticks(rotation=45);

#Exterior1st
plt.figure(figsize=(25,5))
sns.violinplot(x='Exterior1st',y='SalePrice',data=train)
plt.xticks(rotation=45);

# plot discrete numerical variables
fcols = 3
frows = ceil(len(cols_discrete)/fcols)
plt.figure(figsize = (20,frows*4))

for i,col in enumerate(cols_discrete):
    plt.subplot(frows,fcols,i+1)
    sns.violinplot(train[col],train['SalePrice'],innter = "stick")

p_col = dict()

for col in cols_discrete:
    p_col[col] = anova(col,'SalePrice')

pd.Series(p_col).sort_values()
  
# analyze continuous variables

fcols = 2
frows = len(cols_continuous)
plt.figure(figsize=(5*fcols,4*frows))

i=0
for col in cols_continuous:
    i+=1
    ax=plt.subplot(frows,fcols,i)
    sns.regplot(x=col, y='SalePrice', data=train, ax=ax, 
                scatter_kws={'marker':'.','s':3,'alpha':0.3},
                line_kws={'color':'k'});
    plt.xlabel(col)
    plt.ylabel('SalePrice')
    
    i+=1
    ax=plt.subplot(frows,fcols,i)
    sns.distplot(train[col].dropna() , fit=stats.norm)
    plt.xlabel(col)

# create additional columns in test - missing after dummies
y_train = pd.Series(np.log(df_train.SalePrice.values))

X_train,X_valid,y_train,y_valid = train_test_split(train,y_train,test_size=0.25,random_state=1)

# create feature list to throw through lm model - create dynamic list
x_tr = X_train[['TotalSF','OverallQual','KitchenExterQual','GarageCars','NeighGroup','After91','Has2ndFlrSF','HasMiscVal','HasScreenPorch','HasWoodDeckSF','HasOpenPorchSF','HasEnclosedPorch','HasMasVnrArea','HasGarageArea','HasFireplaces','HasTotalBsmtSF','HasVnrStone','HeatingEx','IsDuplex','MSSubClass','Baths','HasPconc','TotRmsAbvGrd','GarageArea','LotFrontage']]
x_tr = pd.get_dummies(x_tr)
x_tr['MSSubClass_150'] = 0

x_val = X_valid[['TotalSF','OverallQual','KitchenExterQual','GarageCars','NeighGroup','After91','Has2ndFlrSF','HasMiscVal','HasScreenPorch','HasWoodDeckSF','HasOpenPorchSF','HasEnclosedPorch','HasMasVnrArea','HasGarageArea','HasFireplaces','HasTotalBsmtSF','HasVnrStone','HeatingEx','IsDuplex','MSSubClass','Baths','HasPconc','TotRmsAbvGrd','GarageArea','LotFrontage']]
x_val = pd.get_dummies(x_val)
x_val['MSSubClass_150'] = 0

x_test = test[['TotalSF','OverallQual','KitchenExterQual','GarageCars','NeighGroup','After91','Has2ndFlrSF','HasMiscVal','HasScreenPorch','HasWoodDeckSF','HasOpenPorchSF','HasEnclosedPorch','HasMasVnrArea','HasGarageArea','HasFireplaces','HasTotalBsmtSF','HasVnrStone','HeatingEx','IsDuplex','MSSubClass','Baths','HasPconc','TotRmsAbvGrd','GarageArea','LotFrontage']]
x_test = pd.get_dummies(x_test)

# find outliers in lasso and remove

x_tr = X_train.copy()
x_tr = pd.get_dummies(x_tr)
outliers = find_outliers(Ridge(),X=x_tr,y=y_train)
x_tr = x_tr.drop(outliers)
y_train = y_train.drop(outliers)

# Start modeling and test multiple models
rmse_scorer = make_scorer(rmse, greater_is_better=False)
opt_models = dict()
score_models = pd.DataFrame(columns=['mean','std'])

# no k-fold splits for train_model function
splits = 5
# no k fold iterations for train_model function
repeats = 5

lr = LinearRegression()
lr.fit(x_tr,y_train)
print_score(lr)

train_sizes,train_scores,test_scores = learning_curve(estimator=lr,X=x_tr,y=y_train,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)

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
plt.ylim([0.0,1.1])
plt.show()

# test ridge regression
model = 'Ridge'
opt_models[model] = Ridge()
param_range = np.arange(0.25,7,0.25)
param_grid = [{'alpha': param_range}]

opt_models[model],cv_score,grid_results = train_model(opt_models[model],X= x_tr,y = y_train,param_grid = param_grid,splits=splits,repeats=repeats)

cv_score.name = model
score_models = score_models.append(cv_score)

en_coefs = pd.Series(opt_models['Ridge'].coef_,index=x_tr.columns)

plt.figure(figsize=(8,20))
en_coefs[en_coefs.abs()>0.02].sort_values().plot.barh()
plt.title('Coefficients with magnitude greater than 0.02')

print('---------------------------------------')
print(sum(en_coefs==0),'zero coefficients')
print(sum(en_coefs!=0),'non-zero coefficients')
print('---------------------------------------')
print('Intercept: ',opt_models['Ridge'].intercept_)
print('---------------------------------------')
print('Top 20 contributers to increased price:')
print('---------------------------------------')
print(en_coefs.sort_values(ascending=False).head(20))
print('---------------------------------------')
print('Top 10 contributers to decreased price:')
print('---------------------------------------')
print(en_coefs.sort_values(ascending=True).head(10))
print('---------------------------------------')
print('Zero coefficients:')
print('---------------------------------------')
print(en_coefs[en_coefs==0].index.sort_values().tolist())

# print alpha value for ridge 
plt.figure()
plt.errorbar(param_range, abs(grid_results['mean_test_score']),
             abs(grid_results['std_test_score'])/np.sqrt(splits*repeats))
plt.xlabel('alpha')
plt.ylabel('score');

ridge_opt = Ridge(alpha=6.75)
ridge_opt.fit(x_tr,y_train)
print_score(ridge_opt)

train_sizes,train_scores,test_scores = learning_curve(estimator=ridge_opt,X=x_tr,y=y_train,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)

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
plt.ylim([0.0,1.1])
plt.show()

# lasso regression
model = 'Lasso'
opt_models[model] = Lasso()
param_range = np.arange(1e-4,0.001,4e-5)
param_grid = [{'alpha': param_range}]

opt_models[model],cv_score,grid_results = train_model(opt_models[model],X= x_tr,y = y_train,param_grid = param_grid,splits=splits,repeats=repeats)

cv_score.name = model
score_models = score_models.append(cv_score)

en_coefs = pd.Series(opt_models['Lasso'].coef_,index=x_tr.columns)

plt.figure(figsize=(8,20))
en_coefs[en_coefs.abs()>0.02].sort_values().plot.barh()
plt.title('Coefficients with magnitude greater than 0.02')

print('---------------------------------------')
print(sum(en_coefs==0),'zero coefficients')
print(sum(en_coefs!=0),'non-zero coefficients')
print('---------------------------------------')
print('Intercept: ',opt_models['Lasso'].intercept_)
print('---------------------------------------')
print('Top 20 contributers to increased price:')
print('---------------------------------------')
print(en_coefs.sort_values(ascending=False).head(20))
print('---------------------------------------')
print('Top 10 contributers to decreased price:')
print('---------------------------------------')
print(en_coefs.sort_values(ascending=True).head(10))
print('---------------------------------------')
print('Zero coefficients:')
print('---------------------------------------')
print(en_coefs[en_coefs==0].index.sort_values().tolist())

#learning curve

Las = Lasso(alpha = 0.00054)
Las.fit(x_tr,y_train)
print_score(Las)

train_sizes,train_scores,test_scores = learning_curve(estimator=Las,X=x_tr,y=y_train,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)

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
plt.ylim([0.0,1.1])
plt.show()

# elastic net

model ='ElasticNet'
opt_models[model] = ElasticNet()

param_grid = {'alpha': np.arange(1e-4,1e-3,1e-4),
              'l1_ratio': np.arange(0.1,1.0,0.1),
              'max_iter':[100000]}


opt_models[model], cv_score, grid_results = train_model(opt_models[model],X=x_tr,y=y_train, param_grid=param_grid, 
            

                                  splits=splits, repeats=repeats)
el = ElasticNet(alpha=0.0009,l1_ratio=0.4)
el.fit(x_tr,y_train)
print_score(el)

cv_score.name = model
score_models = score_models.append(cv_score)

# SVM regressor

model='LinearSVR'
opt_models[model] = LinearSVR()

crange = np.arange(0.1,1.0,0.1)
param_grid = {'C':crange,
             'max_iter':[100000]}

opt_models[model], cv_score, grid_results = train_model(opt_models[model],X=x_tr,y=y_train ,param_grid=param_grid, 
                                              splits=splits, repeats=repeats)

cv_score.name = model
score_models = score_models.append(cv_score)

svmr = LinearSVR(C= 0.9,max_iter=100000,tol=0.0001)
svmr.fit(x_tr,y_train)
print_score(svmr)

# svr

model ='SVR'
opt_models[model] = SVR()

param_grid = {'C':np.arange(1,21,2),
              'kernel':['poly','rbf','sigmoid'],
              'gamma':['auto']}

opt_models[model], cv_score, grid_results = train_model(opt_models[model],X=x_tr,y=y_train, param_grid=param_grid, 
                                              splits=splits, repeats=1)

svr = SVR(C=1,kernel='rbf',gamma='auto')
svr.fit(x_tr,y_train)

cv_score.name = model
score_models = score_models.append(cv_score)

# average predictions
lass_pred = Las.predict(x_test)
ridge_pred = ridge_opt.predict(x_test)
el_pred = el.predict(x_test)
svm_pred = svmr.predict(x_test)
svr_pred = svr.predict(x_test)


ensemble_pred = (np.array(lass_pred) + np.array(ridge_pred) + np.array(el_pred) + np.array(svm_pred) + np.array(svr_pred))/5

# submit pred

current_loc = os.getcwd()

if 'jjonus' in current_loc:
    current_wd = ('C:\\Users\\jjonus\\Google Drive\\Kaggle\\Advanced House Prices')
elif 'jstnj' in current_loc:
    current_wd = ('C:\\Users\\jstnj\\Google Drive\\Kaggle\\Advanced House Prices')
    
test['SalePrice'] = y_pred
submissions = test[['Id','SalePrice']]
submissions['SalePrice'] = np.exp(test['SalePrice'])
submissions['Id'] = submissions['Id'].astype(int)
submissions.to_csv(current_wd + '\\lr_submissions.csv',encoding='utf-8',index=False)

# try stacked prediction

#estlist = [
       # ('lr',LinearRegression()),
        #('KneighborsRegressor',KNeighborsRegressor()),
        # ('GradientBoostingRegressor',GradientBoostingRegressor()),
        # ('metalr',NonNegativeLinearRegression())
        # ]

#sm = StackedRegressor(estlist,n_jobs=-1)
#sm.fit(x,y_train)

