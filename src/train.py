
#coding:utf-8
# train.py
from scipy import stats

from train import  *
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tqdm import tqdm
import xgboost as xgb
from tempfile import mkdtemp
from joblib import Memory
from scipy.sparse import csc_matrix,csr_matrix
import xgboost as xgb
from sklearn.decomposition import TruncatedSVD
import math as mt
import csv
from sparsesvd import sparsesvd
import joblib
import os
import langid
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from fastFM import sgd
import os
import tqdm


X_merged,X_test_merged,y = None,None,None

if os.path.isfile('../out/dump/featuresv2.dump'):
    X_merged, X_test_merged,y = joblib.load('../out/dump/featuresv2.dump')
else:
    X,y,X_test = None,None,None
    if os.path.isfile("../out/dump/preprocessing008.dump"):
        X,y,X_test = joblib.load("../out/dump/preprocessing008.dump")
    else:
        X,y,X_test = preprocessing008()
        joblib.dump((X,y,X_test),filename="../out/dump/preprocessing008.dump",compress=3)

    contexttrain,contexttest = None,None
    if os.path.isfile("../out/dump/preprocessing_source_type_source_screen_name_source_type.dump"):
        contexttrain, contexttest = joblib.load("../out/dump/preprocessing_source_type_source_screen_name_source_type.dump")
    else:
        contexttrain, contexttest = preprocessing_source_type_source_screen_name_source_type()
        joblib.dump((contexttrain,contexttest ),filename="../out/dump/preprocessing_source_type_source_screen_name_source_type.dump",compress=3)


    contexttrain.drop('msno',axis=1,inplace=True)
    contexttest.drop('msno',axis=1,inplace=True)
    contexttrain.fillna(0,inplace=True)
    contexttest.fillna(0,inplace=True)

    #-------------------------
    wordembedding = None
    if os.path.isfile("../out/dump/mpimatrix_galc_svdfeat15allinone.dump"):
        wordembedding = joblib.load("../out/dump/mpimatrix_galc_svdfeat15allinone.dump")
    else:
        pass

    #-------------------------

    #preprocessing_text_apply
    textone,texttwo = None,None
    if os.path.isfile("../out/dump/preprocessing_text.dump"):
        textone,texttwo = joblib.load("../out/dump/preprocessing_text.dump")
    else:
        textone,texttwo = preprocessing_text()
        joblib.dump((textone,texttwo),filename="../out/dump/preprocessing_text.dump",compress=3)

    textone.fillna(-1,inplace=True)
    texttwo.fillna(-1,inplace=True)

    latent = 30
    name = "preprocessing005_dropdup_%d" % latent
    U, V, train, test, train_u, test_u, train_v, test_v = None, None, None, None, None, None, None, None
    if os.path.isfile("../out/dump/%s.dump" % name):
        U, V, train, test, train_u, test_u, train_v, test_v = joblib.load("../out/dump/%s.dump" % name)
    else:
        U, V, train, test, train_u, test_u, train_v, test_v = preprocessing005_dropdup(latent)
        joblib.dump((U, V, train, test, train_u, test_u, train_v, test_v), filename="../out/dump/%s.dump" % name,
                    compress=3)



    X_merged = pd.concat((X, textone, pd.DataFrame(train_u,columns=["SVD_user_%d" % i for i in range(latent)]),
                          pd.DataFrame(train_v, columns=["SVD_item_%d" % i for i in range(latent)]),
                          pd.DataFrame((train_u * train_v), columns=['SVD_useritem_dot_%d' % i for i in range(30)]),
                          pd.DataFrame((train_u * train_v).sum(axis=1),columns=['SVD_useritem_dot']),
                          contexttrain)
                         ,axis=1)

    X_test_merged = pd.concat((X_test, texttwo, pd.DataFrame(test_u,columns=["SVD_user_%d" % i for i in range(latent)]),
                          pd.DataFrame(test_v, columns=["SVD_item_%d" % i for i in range(latent)]),
                               pd.DataFrame((test_u * test_v), columns=['SVD_useritem_dot_%d' % i for i in range(30)])
                               ,pd.DataFrame((test_u * test_v).sum(axis=1),columns=['SVD_useritem_dot']),
                               contexttest)
                         ,axis=1)
    joblib.dump((X_merged, X_test_merged,y),filename='../out/dump/featuresv2.dump',compress=3)

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

wordembedding = None
if os.path.isfile("../out/dump/mpimatrix_galc_svdfeat15allinone.dump"):
    wordembedding = joblib.load("../out/dump/mpimatrix_galc_svdfeat15allinone.dump")
else:
    pass
wordembedding.reset_index(drop=True,inplace=True)
wordembedding.columns = ['wordembedding_%d' % i for i in range(len(wordembedding.columns))]

X_merged = pd.concat((X_merged,wordembedding.iloc[:len(X_merged)]),axis=1)
X_test_merged = pd.concat((X_test_merged,wordembedding.iloc[len(X_merged):].reset_index(drop=True)),axis=1)

print X_merged
print X_test_merged
for newsimi in ['lyricist_split', 'composer_split','artist_name_split' ,'genre_split']:
    genre_simi = pd.read_csv("../out/simi/%s_dot_simi.csv_md" % newsimi,header=None)

    genre_simi.fillna(0,inplace=True)
    X_merged["%s_dot_simi" % newsimi] = genre_simi[0].values[:len(X_merged)]
    X_test_merged["%s_dot_simi" % newsimi] =  genre_simi[0].values[len(X_merged):]

genre_svddot = pd.read_csv("../out/simi/artist_name_svddot.csv",header=None)
X_merged["%s_svddot" % newsimi] = genre_svddot[0].values[:len(X_merged)]
X_test_merged["%s_svddot" % newsimi] =  genre_svddot[0].values[len(X_merged):]

for feature in ['source_system_tab','source_type','source_screen_name']:
    dotsimi = pd.read_csv("../out/simi/%s_dot_simi.csv" % feature,  header=None)
    X_merged["%s_svddot" % feature] = dotsimi[0].values[:len(X_merged)]
    X_test_merged["%s_svddot" % feature] = dotsimi[0].values[len(X_merged):]

cfscore = pd.read_csv('../out/simi/cf_cos_top100.csv',header=None)

X_merged['cfcos100'] = cfscore[0].values[:len(X_merged)]
X_test_merged['cfcos100'] = cfscore[0].values[len(X_merged):]

languagesimi = pd.read_csv('../out/simi/language_dot_simi.csv',header=None)
X_merged['languagesimi'] = languagesimi[0].values[:len(X_merged)]
X_test_merged['languagesimi'] = languagesimi[0].values[len(X_merged):]


# drop song_count features different distribution
selected=[]
X_merged.drop(['song_count'],axis=1,inplace=True)

X_test_merged.drop(['song_count'],axis=1,inplace=True)



#X_merged.drop(['msno','song_id'],axis=1,inplace=True)

#X_test_merged.drop(['msno','song_id'],axis=1,inplace=True)
t
# delete duplicate except target == 1
print "len train before", len(X_merged)
inertest = test
inertrain = train

inerdata = pd.concat((inertrain[['msno', 'song_id']], inertest[['msno', 'song_id']]))
dup = inerdata.duplicated(subset=['msno', 'song_id'], keep='last')
dup = np.logical_not(dup.values)

targetone = (inertrain['target'] == 1)
X_merged['target']=y
X_merged = X_merged[(dup[:len(inertrain)] | targetone ) ]


print "len train after ", len(X_merged),len(y)



print "len drop old user ",len(X_merged)

y = X_merged['target']
X_merged.drop('target',axis=1,inplace=True)
X_merged.reset_index(drop=True,inplace=True)
y.reset_index(drop=True,inplace=True)
X_merged.reset_index(drop=True,inplace=True)


# X_merged['target'] = y
# # X_merged = X_merged.sample(frac=0.1,random_state=111)
# # y = X_merged['target']
# X_merged.drop('target',axis=1,inplace=True)


# X_merged.reset_index(drop=True,inplace=True)
# y.reset_index(drop=True,inplace=True)
#X_merged['target'] = y

# registration_init_time = train['registration_init_time'].cummax()
# train['timestamp'] = registration_init_time
# train.loc[train[train['timestamp']<20161201].index,'timestamp'] = 20161201
# X_merged['timestamp'] = train['timestamp']
# X_merged = X_merged[X_merged['timestamp'] >= 20170101]
# X_merged.drop('timestamp',axis=1,inplace=True)
# X_merged.reset_index(drop=True,inplace=True)

#y = X_merged['target']
#X_merged.drop('target',axis=1,inplace=True)


submit_cv = None
cv=0

dump_name = '12.17add4_dot_simi_addgenrediffsimi_cfcos100score_language_dotsimi_artist_name_svddot_threesimi'



skf = StratifiedKFold(n_splits=10, random_state=253, shuffle=True)

for train_index, valid_index in skf.split(X_merged, y):
    cv += 1
    X_train, X_valid = X_merged.iloc[train_index], X_merged.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]




    submit, validpred, model = lgbm2(X_train, X_valid, y_train, y_valid, X_test_merged, test['id'].values,
                                     params={'learning_rate': 0.1, "application": 'binary', "max_depth": 15,
                                             'num_leaves': 2 ** 8,
                                             'verbosity': 0, 'metric': 'auc', 'num_threads': 50,
                                             'colsample_bytree': 0.9, 'subsample': 0.9}, num_boost_round=3100,
                                     early_stopping_rounds=10)
    if submit_cv is None:
        submit_cv = submit
    else:
        submit_cv['target'] += submit['target']
    submit.to_csv('../out/%s_subcv_%d.csv.gz' % (dump_name, cv), compression='gzip', index=False,
                  float_format='%.8f')
    validpred.to_csv('../out/%s_validcv_%d.csv.gz' % (dump_name, cv), index=False, float_format='%.8f')
    joblib.dump(model,filename='../out/%s_%d.model.dump' % ( dump_name, cv) )

