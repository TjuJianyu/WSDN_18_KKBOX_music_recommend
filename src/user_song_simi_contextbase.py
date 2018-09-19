#features = sys.argv[1]
# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tqdm import tqdm
import xgboost as xgb
from tempfile import mkdtemp
from joblib import Memory
from scipy.sparse import csc_matrix, csr_matrix,lil_matrix
import math as mt
import csv
from sparsesvd import sparsesvd
import joblib
import os
import langid
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import StratifiedKFold
from fastFM import sgd
print('Loading data...')
data_path = "../data/"
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
songs = pd.read_csv(data_path + 'songs.csv')

data = pd.concat((train[['msno', 'song_id', 'target','source_system_tab','source_screen_name','source_type']],
                  test[['msno', 'song_id','source_system_tab','source_screen_name','source_type']]))
data = data.merge(songs[['song_id','language', ]], on='song_id', how='left')


data['language'].fillna(-1,inplace=True)
data['source_system_tab'].fillna('other_source_system_tab',inplace=True)
data['source_screen_name'].fillna('source_screen_name', inplace=True)
data['source_type'].fillna('source_type', inplace=True)

le = LabelEncoder()
data['msno_le'] = msnole = le.fit_transform(data['msno'])
data['song_id_le'] = songle = le.fit_transform(data['song_id'])

def simi_bydump(feature):
    valuecount = data.groupby('msno_le')[feature].value_counts(normalize=False, dropna=False)
    valuecount = valuecount.unstack(level=-1).add_prefix(feature)
    valuecount = valuecount.reset_index(drop=False)
    data_valuecount = data[['msno_le']].merge(valuecount,on='msno_le',how="left")

    data_language = pd.get_dummies(data[feature],prefix=feature)
    data_valuecount.fillna(0,inplace=True)
    data_valuecount.drop('msno_le',axis=1,inplace=True)

    data_valuecount_data_language = (data_valuecount.values - data_language.values)*1.0
    data_valuecount_data_language /= data_valuecount_data_language.sum(axis=1)[:,np.newaxis]


    langsimi = (data_valuecount_data_language * data_language.values).sum(axis=1)
    print pd.Series(langsimi).corr(train['target'])

    pd.DataFrame(langsimi).to_csv("../out/simi/%s_dot_simi.csv" % feature,index=None,header=None)
for feat in ['source_system_tab','source_type','source_screen_name','language']:
    simi_bydump(feat)