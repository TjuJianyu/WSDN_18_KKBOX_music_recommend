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
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
import math as mt
import csv
from sparsesvd import sparsesvd
import joblib
import os
import langid
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import StratifiedKFold
from fastFM import sgd

from multiprocessing import Pool
import sys
from sklearn.metrics import pairwise_distances

njobs = int(sys.argv[1])
print('Loading data...')
data_path = "../data/"
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')

print('Data preprocessing merge and fillna and numeric encoding...')
data = pd.concat((train[['msno', 'song_id', 'target']], test[['msno', 'song_id']]))
le = LabelEncoder()
data['msno_le'] = msnole = le.fit_transform(data['msno'])
data['song_id_le'] = songle = le.fit_transform(data['song_id'])
data_nodup = data.drop_duplicates(['msno_le','song_id_le'])

uimatrix = csr_matrix(([1]*len(data_nodup),(data_nodup['msno_le'].values,data_nodup['song_id_le'].values)))

print "construct similarity matrix"
uicossimi = pairwise_distances(uimatrix,metric='cosine',n_jobs=60)
uicossimi = 1-uicossimi
uicossimi -= np.eye(uicossimi.shape[0])
uicossimi[uicossimi<1e-10]=0

print "sort simi"
argsortindex = np.argsort(uicossimi)


#
iumatrix = uimatrix.transpose()
iumatrix = iumatrix.tolil()



for topn in [100]:
    index = []
    for i in range(uicossimi.shape[0]):
        index.extend([i]*topn)
    maskmatrix = csr_matrix(([1]*topn*uicossimi.shape[0],(index,argsortindex[:,-topn:].flatten())))
    uicossimiloc =(maskmatrix).multiply(uicossimi).tolil()


    def cfscore(input):
        uu_similoc = uicossimiloc[input[0]].tocsr().tocoo()
        iu_act = iumatrix[input[1]].tocsr().tocoo()
        return np.array(uu_similoc.multiply(iu_act).sum(axis=1)/topn).flatten()


    p = Pool(njobs)
    para = []
    for i in range(max(njobs-1,10)):
        para.append((data['msno_le'].values[len(data)/njobs*(i):len(data)/njobs*(i+1)]
        ,data['song_id_le'].values[len(data)/njobs*(i):len(data)/njobs*(i+1)]))

    para.append((data['msno_le'].values[len(data)/njobs*(njobs-1):]
        ,data['song_id_le'].values[len(data)/njobs*(njobs-1):]))



    simi = p.map(cfscore, para)
    simiflatten = []
    for val in simi:
        simiflatten.extend(val)

    #joblib.dump(simiflatten,filename="../out/simi/cf_cos_top%d.dump" % topn)
    simi_pd = pd.DataFrame(simiflatten)
    simi_pd.to_csv("../out/simi/cf_cos_top%d.csv" % topn,index=None,header=None)
