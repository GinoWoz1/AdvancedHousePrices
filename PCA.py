
from sklearn.model_selection import learning_curve,validation_curve,cross_val_score,GridSearchCV,train_test_split,RepeatedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# start modelling - check the learning and validation curve along the way  
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import Lasso,Ridge,ElasticNet
from fastai.imports import *  # fast ai library
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator, ZeroCount
from sklearn.svm import LinearSVR,SVR
from sklearn.decomposition import PCA
from scipy.stats import norm, skew
from tpot import TPOTRegressor   
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from pandas_summary import * 
from scipy.stats import norm
from fancyimpute import MICE
import missingno as msno
from scipy import stats
import xgboost as xgb
import seaborn as sns
from math import ceil
import pandas as pd
import numpy as np
import warnings
import os


warnings.filterwarnings('ignore')
%matplotlib inline

# score functions
def learning_c(X_train,y_train):
    train_sizes,train_scores,test_scores = learning_curve(estimator=exported_pipeline,X=X_train,y=y_train,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)
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
    
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(x_tr),y_train),rmse(m.predict(x_val),y_valid),m.score(x_tr,y_train),m.score(x_val,y_valid)]
    print(res)
    
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

# read data in

components = 50
def pca_processing(components):
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
    
    # run imputation separately on each data set
    numeric_data_train = all_data.select_dtypes(include = ['float64','int64'])
    solver=MICE()
    Imputed_dataframe_train= pd.DataFrame(data = solver.complete(numeric_data_train),columns = numeric_data_train.columns,index = numeric_data_train.index)

    # set back to main dataframe dataset
    all_data['LotFrontage'] = Imputed_dataframe_train['LotFrontage']

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
    all_data[scale_features] = scaler.fit_transform(all_data[scale_features])
    pca = PCA(n_components = components)
    all_data = pd.get_dummies(all_data)
    all_pca = pca.fit_transform(all_data)
    train = all_pca[:ntrain]
    test = all_pca[ntrain:]
    y_train = pd.Series(np.log(df_train.SalePrice.values))
    X_train,X_valid,y_train,y_valid = train_test_split(train,y_train,test_size=0.25,random_state=1)
    def rmsle_loss(y_true, y_pred):
        assert len(y_true) == len(y_pred)
        terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y_true[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
        if not (y_true >= 0).all() and not (y_pred >= 0).all():
            raise ValueError("Mean Squared Logarithmic Error cannot be used when "
                             "targets contain negative values.")
        return (sum(terms_to_sum) * (1.0/len(y_true))) ** 0.5
    
    return X_train,X_valid,y_train,y_valid


def rmsle_loss(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y_true[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    if not (y_true >= 0).all() and not (y_pred >= 0).all():
            raise ValueError("Mean Squared Logarithmic Error cannot be used when "
                             "targets contain negative values.")
    return (sum(terms_to_sum) * (1.0/len(y_true))) ** 0.5

rmsle_loss = make_scorer(rmsle_loss,greater_is_better=False)   

X_train,X_valid,y_train,y_valid = pca_processing(25)

tpot = TPOTRegressor(verbosity=3,scoring=rmsle_loss,population_size=50,offspring_size= 50,periodic_checkpoint_folder='C:\\Users\\jstnj\\Google Drive\\Kaggle\\Advanced House Prices',max_eval_time_mins=10,warm_start=True)
tpot.fit(X_train,y_train)

y_pred = exported_pipeline.predict(test)
test_new = pd.DataFrame(test)
test_new['SalePrice'] = y_pred
test_new['Id'] = df_test['Id']
submissions = test_new[['Id','SalePrice']]
submissions['SalePrice'] = np.exp(test_new['SalePrice'])
submissions['Id'] = submissions['Id'].astype(int)
submissions.to_csv(current_wd + '\\lr_submissions.csv',encoding='utf-8',index=False)