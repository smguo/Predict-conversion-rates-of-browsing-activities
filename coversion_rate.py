
#get_ipython().magic('matplotlib inline')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import (GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # Used for imputing rare / missing values

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression # only model used for final submission

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from operator import itemgetter

os.chdir('E:\\Google Drive\\Python\\DS challenges')        


#%% read train and test data
def read_table():
    print("Read conversion_data.csv...")
    df = pd.read_csv("./conversion_data.csv",
                       dtype={'country': np.str,
                              'source': np.str,
                              'age': np.int32,
                              'new_user':np.int8,
                              'converted':np.int8,
                              'total_pages_visited': np.int32})
    
    return df                   
#%%        
def feature_engineer(train):
    print('Construct features...')
    
    categorical=['country','source']        
    train = pd.get_dummies(train, prefix = categorical)
    
    return train

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance
#%%    
def tune_cls_para(alg, train, features, target, param_test, n_jobs=8):
    
    gsearch = GridSearchCV(estimator = alg,param_grid = param_test, 
        scoring='roc_auc',n_jobs=n_jobs,iid=False, verbose=3, 
        cv=StratifiedKFold(train[target], n_folds=5, shuffle=True))
    gsearch.fit(train[features],train[target]) 
    alg.set_params(**gsearch.best_params_)
    for score in gsearch.grid_scores_:
        print(score)         
    print('best CV parameters:')
    print(gsearch.best_params_)
    print('best CV score: %f' % gsearch.best_score_)    
    return alg
#%%
def xgb_fit(alg, dtrain, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, stratified=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
#    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
        
    #Predict training set:
#    dtrain_predictions = alg.predict(dtrain[predictors])
#    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print("\nModel Report")
    print('# of estimators: %d' % cvresult.shape[0])
#    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Test): %f" % cvresult['test-auc-mean'].iloc[-1])
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    return alg

#%% main script 
df = read_table()
df.describe() #summerize the table
df = df[df.age<80] # remove outliers
#%% visualize the data 
plt.figure(1, figsize=(8,6))
ax = sns.countplot(x="country", data=df, linewidth=1)
plt.figure(2, figsize=(8,6))
ax = sns.barplot(x="country", y='converted', data=df,
                     ci=95, linewidth=1, errwidth=1, capsize=0.15, errcolor =[0,0,0])
plt.figure(3, figsize=(8,6))
ax = sns.pointplot(x="age", y='converted', data=df,markers="",
                     ci=95, linewidth=1, errwidth=1, capsize=0.15, errcolor =[0,0,0])
plt.figure(4, figsize=(8,6))
ax = sns.barplot(x="new_user", y='converted', data=df,
                     ci=95, linewidth=1, errwidth=1, capsize=0.15, errcolor =[0,0,0])
plt.figure(5, figsize=(8,6))
ax = sns.barplot(x="source", y='converted', data=df,
                     ci=95, linewidth=1, errwidth=1, capsize=0.15, errcolor =[0,0,0])
plt.figure(6, figsize=(8,6))
ax = sns.pointplot(x="total_pages_visited", y="converted", data=df,markers="")

#%%
train = df
train = feature_engineer(train)
features = ['age', 'new_user', 'total_pages_visited', 'country_China',
       'country_Germany', 'country_UK', 'country_US', 'source_Ads',
       'source_Direct', 'source_Seo']
# train leak model using only group_1 and activity_date info
print('Length of train: ', len(train))

print('Features [{}]: {}'.format(len(features), sorted(features)))

#%% define XGB classifier
xgb1 = XGBClassifier(
 learning_rate=0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=8,
 scale_pos_weight=1,
 seed=27)
xgb1 = xgb_fit(xgb1, train, features, 'converted')
#%%
param_test0 = {'scale_pos_weight': 2**np.arange(0,7,1)}
xgb1 = tune_cls_para(xgb1, train, features, 'converted', param_test0)
#%%
param_test1 = {
    'max_depth':np.arange(3,10,2),
     'min_child_weight':np.arange(1,6,2)
    }
xgb1 = tune_cls_para(xgb1, train, features, 'converted', param_test1)
#%%
param_test2 = {
    'max_depth':np.arange(1,4,1),
     'min_child_weight':np.arange(0,1.2,0.2)
    }
xgb1 = tune_cls_para(xgb1, train, features, 'converted', param_test2)

#%%
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
xgb1 = tune_cls_para(xgb1, train, features, 'converted', param_test3)
#%%
xgb1.set_params(n_estimators=1000)
xgb1 = xgb_fit(xgb1, train, features, 'converted')
#%%
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
xgb1 = tune_cls_para(xgb1, train, features, 'converted', param_test4)
#%%
param_test6 = {
 'reg_alpha':[0]+[10**i for i in range(-2, 6, 2)],
 'reg_lambda':[0]+[10**i for i in range(-2, 6, 2)]
}
xgb1 = tune_cls_para(xgb1, train, features, 'converted', param_test6)

#%%
xgb1.set_params(learning_rate=0.01, n_estimators=1000)
xgb1 = xgb_fit(xgb1, train, features, 'converted')
#%% SVM classifier
# RBF kernel SVC doesn't scale to larger sample size well. Use linear SVC instead
scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
C_range = np.logspace(-2, 10, 6)
gamma_range = np.logspace(-9, 3, 6)
#param_SVC = dict(gamma=gamma_range, C=C_range)
param_SVC = dict(C=C_range)
LSVC1 = tune_cls_para(LinearSVC(), train, features, 'converted', param_SVC, n_jobs=20)
#best CV score: 0.986020
#%% Random forest classifier
RFC1 = RandomForestClassifier(n_jobs=2)
#mf_range = np.linspace(0.1, 1, 5)
mf_range = np.arange(1,11,2, dtype=int)
n_est_range = [1000, 4000, 100000]
param_RFC = dict(max_features=mf_range, n_estimators = n_est_range)
RFC1 = tune_cls_para(RFC1, train, features, 'converted', param_RFC, n_jobs=10)

#%% Logistic regression
scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
LGR1=LogisticRegression(C=1.0, intercept_scaling=1, dual=False, fit_intercept=True,
          penalty='l2', tol=0.0001)
C_range = np.logspace(-2, 5, 4)
class_weight_range = [{1: w} for w in 2**np.arange(0,13,2)]
param_LGR = dict(C=C_range, class_weight=class_weight_range)
LGR1 = tune_cls_para(LGR1, train, features, 'converted', param_LGR, n_jobs=10)

