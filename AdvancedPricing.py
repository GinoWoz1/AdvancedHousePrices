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
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
import missingno as msno
from fancyimpute import MICE
from scipy import stats
from scipy.stats import norm, skew

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

# save id column

#train_ID = df_train['Id']
#test_ID = df_test['Id']

# feature engineering - combine data sets first

ntrain = df_train.shape[0]
ntest = df_test.shape[0]
y_train = np.log(df_train.SalePrice.values)
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

# boxplot lot frontage by neighborhood

plt.figure(figsize=(20,15))
sns.boxplot(x = "Neighborhood",y="LotFrontage",data = all_data)
plt.show()

# analyze which neighborhoods have null data

all_data[pd.isna(all_data['LotFrontage'])].groupby("Neighborhood").size().sort_values(ascending = False)

#impute lot frontage and other empty values

numeric_data = all_data.select_dtypes(include = ['float64','int64'])
solver=MICE()
Imputed_dataframe= pd.DataFrame(data = solver.complete(numeric_data),columns = numeric_data.columns,index = numeric_data.index)
all_data['LotFrontage'] = Imputed_dataframe['LotFrontage']

# change from number to categorical variable (set in string)

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

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
skewness = skewness[skewness.feature != 'TotRmsAbvGrd']
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

scaler = StandardScaler()
scale_features = skewness.feature.values.tolist()

all_data[scale_features] = scaler.fit_transform(all_data[scale_features])

# change all variables to float and check remaining data types

cols = all_data.loc[:,all_data.dtypes == np.int64].columns.tolist()
all_data[cols] = all_data[cols].astype(float)
all_data.columns.to_series().groupby(all_data.dtypes).groups

# make dummy columns and change remaining 

#all_data = pd.get_dummies(all_data)
#all_data['Id'] = all_data['Id'].astype(str)
#all_data['YearBuilt'] = all_data['YearBuilt'].astype(str)
#all_data['YearRemodAdd'] = all_data['YearRemodAdd'].astype(str)
#all_data['GarageYrBlt'] = all_data['GarageYrBlt'].astype(str)

# split back into train and test

train = all_data[:ntrain]
test = all_data[ntrain:]

# start modelling - check the learning and validation curve along the way
    
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve,validation_curve,cross_val_score
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from statsmodels.tools.tools import add_constant
from civismlext.stacking import StackedRegressor
from civismlext.nonnegative import NonNegativeLinearRegression

# rename columns and create func for changing housestyle values

def func(row):
    if row['HouseStyle'] == '2Story':
        return 'TwoStory'
    elif row['HouseStyle'] =='1Story':
        return 'OneStory' 
    elif row['HouseStyle'] =='1.5Fin':
        return 'OneFiveFin' 
    elif row['HouseStyle'] =='1.5Unf':
        return 'OneFivUnf' 
    elif row['HouseStyle']=='2.5Unf':
        return 'TwoFiveUnf'
    elif row['HouseStyle']=='2.5Fin':
        return 'TwoFiveFin' 
    else:
        return row['HouseStyle']

train['HouseStyle'] = train.apply(func,axis=1)
train.rename(columns = {'1stFlrSF':'FirstFlrSF','2ndFlrSF':'SecondFlrSF'},inplace=True)
train['SalePrice'] = y_train

test['HouseStyle'] = test.apply(func,axis=1)
test.rename(columns = {'1stFlrSF':'FirstFlrSF','2ndFlrSF':'SecondFlrSF'},inplace=True)

# create feature for 1st or 2nd floor

train['FirstOrSecond'] = np.where(train['SecondFlrSF'] < -0.70 ,"FirstFloor","SecondFloorAbove")
train['HalfBathTransformed'] = train['HalfBath'] * 0.5
train['HalfBathBsmtTransformed'] = train['BsmtHalfBath'] * 0.5
train['Baths'] = train['FullBath'] + train['HalfBathTransformed'] + train['HalfBathBsmtTransformed']  + train['BsmtFullBath']
train['MSZoning'] = np.where(train['MSZoning'] == 'C (all)','MSZoning_C',train['MSZoning'])


test['FirstOrSecond'] = np.where(test['SecondFlrSF'] < -0.70 ,"FirstFloor","SecondFloorAbove")
test['HalfBathTransformed'] = test['HalfBath'] * 0.5
test['HalfBathBsmtTransformed'] = test['BsmtHalfBath'] * 0.5
test['Baths'] = test['FullBath'] + test['HalfBathTransformed'] + test['HalfBathBsmtTransformed']  + test['BsmtFullBath']
test['MSZoning'] = np.where(test['MSZoning'] == 'C (all)','MSZoning_C',test['MSZoning'])

# create feature list to throw through lm model - create dynamic list
x_val = train[['GrLivArea','OverallQual','BsmtQual','ExterQual','FirstOrSecond','SalePrice','Foundation','GarageCars','GarageFinish','Neighborhood','MSZoning','Baths','HouseStyle']]
x_val = pd.get_dummies(x_val)

x_val_test = test[['GrLivArea','OverallQual','BsmtQual','ExterQual','FirstOrSecond','Foundation','GarageCars','GarageFinish','Neighborhood','MSZoning','Baths','HouseStyle']]
x_val_test = pd.get_dummies(x_val_test)
x_val_test['HouseStyle_TwoFiveFin'] = 0

features = " + ".join(x_val.columns).replace("SalePrice","")

lm = smf.ols(formula = 'SalePrice ~' + features ,data = x_val).fit()

print(lm.summary())

y_pred_test = lm.predict(x_val_test).tolist()

# check multi-collinearity

y, X = dmatrices('SalePrice ~' + features, x_val, return_type='dataframe')
X = add_constant(X)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)

# test learning curve

x = train[['GrLivArea','TotRmsAbvGrd','OverallQual','BsmtQual','ExterQual','FirstOrSecond','Foundation','GarageCars','GarageFinish','Neighborhood','MSZoning','Baths','HouseStyle']]
x = pd.get_dummies(x)
lr = LinearRegression()
lr.fit(x,y_train)

train_sizes,train_scores,test_scores = learning_curve(estimator=lr,X=x,y=y_train,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)

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

# submit pred

test['SalePrice'] = y_pred_test

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

y_pred = lm.predict(x_val_test)

test['SalePrice'] = y_pred

submissions = test[['Id','SalePrice']]
submissions['SalePrice'] = np.exp(test['SalePrice'])
submissions['Id'] = submissions['Id'].astype(int)
submissions.to_csv(current_wd + '\\lr_submissions.csv',encoding='utf-8',index=False)

# attempt to use random forest to choose best features.

from sklearn.model_selection import GridSearchCV

x_val.drop(['SalePrice'],axis=1,inplace=True)

min_leaf = [1,2,3,5,6]
estimators = [50,100,200,300,400,500]
max_features = ['sqrt','log2',0.5,0.2]

rf = RandomForestRegressor()

param_grid = [{'n_estimators': estimators,
               'max_features': max_features,
               'min_samples_leaf':min_leaf}]

gs = GridSearchCV(estimator = rf,param_grid=param_grid,
                  scoring = 'neg_mean_squared_error',
                  cv=10,n_jobs=1)

gs = gs.fit(x_val,y_train)
gs.best_score_
gs.best_params_

# build model based off of best params

from fastai.imports import *

x_val.drop(['SalePrice'],axis=1,inplace=True)
rf = RandomForestRegressor(max_features = 'sqrt',min_samples_leaf = 3, n_estimators = 300,n_jobs =-1,oob_score = True)
rf.fit(x_val,y_train)

# check if highly correlated

from scipy.cluster import hierarchy as hc
from scipy import stats

feat_labels = x_val.columns
importances = rf.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range (x_val.shape[1]):
    print("%2d)%-*s %f" % (f + 1,30,
         feat_labels[indices[f]],
         importances[indices[f]]))

# predict with significant variables

df_imp = pd.DataFrame(importances,columns = ['importance'])
df_imp['Label'] = feat_labels
df_new = df_imp[df_imp['importance'] > 0.005]

columns = df_new.Label.tolist()
x_val = x_val[columns]
x_val['SalePrice'] = y_train

x_val_test = x_val_test[columns]

features = " + ".join(x_val.columns).replace("+ SalePrice","")

lm = smf.ols(formula = 'SalePrice ~ ' + features ,data = x_val).fit()

print(lm.summary())

# see where highly correlated

x_val.drop(['SalePrice'],axis=1,inplace=True)

corr = np.round(stats.spearmanr(x_val).correlation,4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed,method='average')
fig = plt.figure(figsize =(16,10))
dendrogram = hc.dendrogram(z,labels= x_val.columns,orientation = 'left',leaf_font_size =8)
plt.show()

# test learning curve now

lr = LinearRegression()
lr.fit(x_val,y_train)

train_sizes,train_scores,test_scores = learning_curve(estimator=lr,X=x_val,y=y_train,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)

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

# send predictions

y_pred = lm.predict(x_val_test)

test['SalePrice'] = y_pred_2

submissions = test[['Id','SalePrice']]
submissions['SalePrice'] = np.exp(test['SalePrice'])
submissions['Id'] = submissions['Id'].astype(int)
submissions.to_csv(current_wd + '\\lr_submissions.csv',encoding='utf-8',index=False)

# test leverage

influence = lm.get_influence()
resid_student = influence.resid_studentized_external
(cooks, p) = influence.cooks_distance
(dffits, p) = influence.dffits
leverage = influence.hat_matrix_diag

print('\n')
print('Leverage v.s. Studentized Residuals')
sns.regplot(leverage, lm.resid_pearson,  fit_reg=False)   


