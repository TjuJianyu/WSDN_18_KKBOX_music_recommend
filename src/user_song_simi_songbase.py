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
import sys
import numpy as np
from itertools import product
import scipy.sparse as sparse
import scipy

from multiprocessing import Pool
def uifeat_generate(transection):
    data = transection
    user_itemfeat = data.groupby('msno')[['msno_le']].first()

    print 'genreids...'
    user_itemfeat['genre_ids'] = data.groupby('msno')['genre_ids'].agg(lambda x: "|".join(x))
    print 'artist...'
    user_itemfeat['artist_name'] = data.groupby('msno')['artist_name'].agg(lambda x: "|".join(x))
    print 'lyricist...'
    user_itemfeat['lyricist'] = data.groupby('msno')['lyricist'].agg(lambda x: "|".join(x))
    print 'composer...'
    user_itemfeat['composer'] = data.groupby('msno')['composer'].agg(lambda x: "|".join(x))

    print 'genreids...'
    user_itemfeat['genre_split'] = user_itemfeat.genre_ids.str.split(r'[;|/\\,+、&]+')
    print 'artist_name...'
    user_itemfeat['artist_name_split'] = user_itemfeat.artist_name.str.split(r',|\||\\|&|\+|;|、| ，|and|/|feat.|features|featuring|with| X |')
    print 'lyricist...'
    user_itemfeat['lyricist_split'] = user_itemfeat.lyricist.str.split(r',|\||\\|&|\+|;|、| ，|and|/|feat.|features|featuring|with| X |')
    print 'composer...'
    user_itemfeat['composer_split'] = user_itemfeat.composer.str.split(r',|\||\\|&|\+|;|、| ，|and|/|feat.|features|featuring|with| X |')
    print "user reset index"
    user_itemfeat.reset_index(drop=False, inplace=True)
    return user_itemfeat


def songfeat_generate(subsongs):
    print "song sink"
    songs = subsongs.copy()
    songs['genre_ids'].fillna('others_genre', inplace=True)
    songs['artist_name'].fillna('other_artist', inplace=True)
    songs['composer'].fillna('other_composer', inplace=True)
    songs['lyricist'].fillna('other_lyricist', inplace=True)
    songs['genre_split'] = songs.genre_ids.str.split(r'[;|/\\,+、&]+')
    print 'artist_name...'
    songs['artist_name_split'] = songs.artist_name.str.split(r',|\||\\|&|\+|;|、| ，|and|/|feat.|features|featuring|with| X |')
    print 'lyricist...'
    songs['lyricist_split'] = songs.lyricist.str.split(r',|\||\\|&|\+|;|、| ，|and|/|feat.|features|featuring|with| X |')
    print 'composer...'
    songs['composer_split'] = songs.composer.str.split(r',|\||\\|&|\+|;|、| ，|and|/|feat.|features|featuring|with| X |')
    return songs


def useritemfeat_vec(userfeat, songfeat, urows_matrix=None, srows_matrix=None, features='genre_split'):
    # for features in ['genre_split', 'lyricist_split', 'composer_split','artist_name_split' ]:
    le = LabelEncoder()
    user_itemfeat = userfeat
    songs = songfeat
    print 'vec for item feat'
    index = []
    value = []

    farray = songs[['song_id_le', features]].values
    for i in range(len(farray)):
        uid = farray[i][0]
        val = farray[i][1]
        value.extend(val)
        index.extend([uid] * len(val))
    le.fit(value)
    item_genreid_le = le.transform(value)

    shapeu = (userfeat['msno_le'].max() + 1, item_genreid_le.max() + 1)
    shapes = (songfeat['song_id_le'].max() + 1, item_genreid_le.max() + 1)
    if urows_matrix is not None:
        shapeu = (urows_matrix, item_genreid_le.max() + 1)
    if srows_matrix is not None:
        shapes = (srows_matrix, item_genreid_le.max() + 1)

    item_genreid_matrix = csr_matrix(([1.0] * len(index), (index, item_genreid_le)), shape=shapes).tolil()
    print "vec for item feat done"
    print "vec for user feat"
    index = []
    value = []
    farray = user_itemfeat[['msno_le', features]].values
    for i in range(len(farray)):
        mid = farray[i][0]
        val = farray[i][1]
        value.extend(val)
        index.extend([mid] * len(val))
    print 'label transform...'
    user_feat_le = le.transform(value)
    usergenre_matrix = csr_matrix(([1.0] * len(index), (index, user_feat_le)), shape=shapeu).tolil()
    print "done"
    return usergenre_matrix, item_genreid_matrix

def similarity():

if __name__ == "__main__":
    #features = sys.argv[1]
    njobs = int(sys.argv[1])
    print('Loading data...')
    data_path = "../data/"
    train = pd.read_csv(data_path + 'train.csv')
    test = pd.read_csv(data_path + 'test.csv')
    songs = pd.read_csv(data_path + 'songs.csv')
    members = pd.read_csv(data_path + 'members.csv')

    for col in ['artist_name', 'composer', 'lyricist']:
        songs[col] = songs[col].str.replace(r'\([\s\S]+?\)', '')
        songs[col] = songs[col].str.replace(r'（[\s\S]+?）', '')

    print('Data preprocessing merge and fillna and numeric encoding...')
    data = pd.concat((train[['msno', 'song_id', 'target']], test[['msno', 'song_id']]))
    data = data.merge(songs[['song_id', 'genre_ids', 'artist_name', 'composer', 'lyricist']], on='song_id', how='left')
    data['genre_ids'].fillna("others_genre", inplace=True)
    data['artist_name'].fillna('other_artist', inplace=True)
    data['composer'].fillna('other_composer', inplace=True)
    data['lyricist'].fillna('other_lyricist', inplace=True)
    data.ix[data[(data['composer'].str.len() > 500)].index, 'composer'] = 'other_composer'
    le = LabelEncoder()
    data['msno_le'] = msnole = le.fit_transform(data['msno'])
    data['song_id_le'] = songle = le.fit_transform(data['song_id'])
    print data['msno_le'].max()
    print('Data preprocessing songs...')
    songs.ix[songs[songs['composer'].str.len() > 500].index, 'composer'] = 'other_composer'
    songs = songs.merge(data[['song_id', 'song_id_le']].drop_duplicates(), on='song_id', how='right')

    ufeat = uifeat_generate(data)
    ifeat = songfeat_generate(songs)

    for features in [ 'lyricist_split', 'composer_split','artist_name_split' ,'genre_split']:
        ufeatmatrix, ifeatmatrix = useritemfeat_vec(ufeat, ifeat,features=features)


        def similarity_vec_njobs(input):

            a = ufeatmatrix[input[0]].tocsr().tocoo()
            b = ifeatmatrix[input[1]].tocsr().tocoo()
            #print "simi"
            a_b = a - b
            #print "1/3"
            a_b = a_b.multiply(1 / a_b.sum(axis=1))
            #print "2/3"
            b_ = b.multiply(1 / b.sum(axis=1))
            #print "3/3"
            cossimi = a_b.multiply(b_).sum(
                axis=1).A1  # / ((a_b.multiply(a_b)).sum(axis=1).A1 * (b_.multiply(b_)).sum(axis=1)).sum(axis=1).A1
            #print "done"
            return cossimi


        p = Pool(njobs)
        para = []
        for i in range(max(njobs-1,10)):
            para.append((data['msno_le'].values[len(data)/njobs*(i):len(data)/njobs*(i+1)]
            ,data['song_id_le'].values[len(data)/njobs*(i):len(data)/njobs*(i+1)]))

        para.append((data['msno_le'].values[len(data)/njobs*(njobs-1):]
            ,data['song_id_le'].values[len(data)/njobs*(njobs-1):]))

        print 'compute simi'
        simi = p.map(similarity_vec_njobs,para)
        simiflatten = []
        for val in simi:
            simiflatten.extend(val)
        simi_pd = pd.DataFrame(simiflatten)
        simi_pd.to_csv('../out/simi/%s_dot_simi_clean.csv'  % features,index=None)
