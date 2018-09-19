#coding:utf-8
# train.py
import numpy as np
import pandas as pd

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
from fastFM import sgd
import os
import xgboost as xgb

def preprocessing001():
	print('Loading data...')
	data_path = "../data/"
	train = pd.read_csv(data_path + 'train.csv')
	test = pd.read_csv(data_path + 'test.csv')
	songs = pd.read_csv(data_path + 'songs.csv')
	members = pd.read_csv(data_path + 'members.csv')
	print('Data preprocessing...')

	song_cols = songs.columns #['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']
	train = train.merge(songs[song_cols], on='song_id', how='left')
	test = test.merge(songs[song_cols], on='song_id', how='left')

	members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
	members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
	members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

	members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
	members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
	members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
	members = members.drop(['registration_init_time'], axis=1)

	members_cols = members.columns
	train = train.merge(members[members_cols], on='msno', how='left')
	test = test.merge(members[members_cols], on='msno', how='left')

	train = train.fillna(-1)
	test = test.fillna(-1)
	import gc
	del members, songs;
	gc.collect();

	cols = list(train.columns)
	cols.remove('target')

	for col in tqdm(cols):
	    if train[col].dtype == 'object':
	        train[col] = train[col].apply(str)
	        test[col] = test[col].apply(str)
	        le = LabelEncoder()
	        train_vals = list(train[col].unique())
	        test_vals = list(test[col].unique())
	        le.fit(train_vals + test_vals)
	        train[col] = le.transform(train[col])
	        test[col] = le.transform(test[col])

	X = np.array(train.drop(['target'], axis=1))
	y = train['target'].values

	X_test = np.array(test.drop(['id'], axis=1))
	ids = test['id'].values
	return X,y,X_test

def preprocessing_text():
	print('Loading data...')
	data_path = "../data/"
	train = pd.read_csv(data_path + 'train.csv')
	test = pd.read_csv(data_path + 'test.csv')
	songs = pd.read_csv(data_path + 'songs.csv')
	members = pd.read_csv(data_path + 'members.csv')
	songsinfo = pd.read_csv(data_path + "song_extra_info.csv")
	print('Data preprocessing...')

	df_train_merged = train.merge(songs[['song_id', 'genre_ids', 'artist_name', 'composer', 'lyricist']], on='song_id',
								  how='left')
	df_test_merged = test.merge(songs[['song_id', 'genre_ids', 'artist_name', 'composer', 'lyricist']], on='song_id',
								how='left')

	df_train_merged[['genre_ids', 'artist_name', 'composer', 'lyricist']].fillna('', inplace=True)
	df_test_merged[['genre_ids', 'artist_name', 'composer', 'lyricist']].fillna('', inplace=True)

	df_train_merged['genre_ids'] = df_train_merged['genre_ids'].astype(str)
	df_test_merged['genre_ids'] = df_test_merged['genre_ids'].astype(str)
	df_train_merged['artist_name'] = df_train_merged['artist_name'].astype(str)
	df_test_merged['artist_name'] = df_test_merged['artist_name'].astype(str)
	df_train_merged['composer'] = df_train_merged['composer'].astype(str)
	df_test_merged['composer'] = df_test_merged['composer'].astype(str)
	df_train_merged['lyricist'] = df_train_merged['lyricist'].astype(str)
	df_test_merged['lyricist'] = df_test_merged['lyricist'].astype(str)

	df_train_merged['genre_ids_len'] = df_train_merged.genre_ids.str.split(r'|').apply(lambda x: len(x))
	df_test_merged['genre_ids_len'] = df_test_merged.genre_ids.str.split(r'|').apply(lambda x: len(x))

	df_train_merged['artist_name_len'] = df_train_merged.artist_name.str.split(r'[;|/\\,+、&]+').apply(lambda x: len(x))
	df_test_merged['artist_name_len'] = df_test_merged.artist_name.str.split(r'[;|/\\,+、&]+').apply(lambda x: len(x))

	df_train_merged['composer_len'] = df_train_merged.composer.str.split(r'[;|/\\,+、&]+').apply(lambda x: len(x))
	df_test_merged['composer_len'] = df_test_merged.composer.str.split(r'[;|/\\,+、&]+').apply(lambda x: len(x))

	df_train_merged['lyricist_len'] = df_train_merged.lyricist.str.split(r'[;|/\\,+、&]+').apply(lambda x: len(x))
	df_test_merged['lyricist_len'] = df_test_merged.lyricist.str.split(r'[;|/\\,+、&]+').apply(lambda x: len(x))

	def findfeat(merged):
		feat = merged['artist_name'].str.find('feat.')
		feat[feat == -1] = 0
		feat[feat > 0] = 1
		feat2 = merged['artist_name'].str.find('featuring')
		feat2[feat2 == -1] = 0
		feat2[feat2 > 0] = 1
		feat3 = feat + feat2
		feat3[feat3 > 0] = 1
		return feat3

	df_train_merged['is_featured'] = findfeat(df_train_merged)
	df_test_merged['is_featured'] = findfeat(df_test_merged)

	df_train_merged['artist_composer'] = (df_train_merged['artist_name'] == df_train_merged['composer']).astype(np.int8)
	df_test_merged['artist_composer'] = (df_test_merged['artist_name'] == df_test_merged['composer']).astype(np.int8)

	df_train_merged['artist_composer_lyricist'] = ((df_train_merged['artist_name'] == df_train_merged['composer']) & (
		df_train_merged['artist_name'] == df_train_merged['lyricist']) & (
													   df_train_merged['composer'] == df_train_merged[
														   'lyricist'])).astype(
		np.int8)
	df_test_merged['artist_composer_lyricist'] = ((df_test_merged['artist_name'] == df_test_merged['composer']) & (
		df_test_merged['artist_name'] == df_test_merged['lyricist']) & (
													  df_test_merged['composer'] == df_test_merged['lyricist'])).astype(
		np.int8)

	data = pd.concat((df_train_merged, df_test_merged))
	songcount = data.groupby('song_id')[['msno']].count()
	songcount.reset_index(inplace=True)
	songcount.rename(columns={'msno': 'msno_count'}, inplace=True)

	usercount = data.groupby('msno')[['song_id']].count()
	usercount.reset_index(inplace=True)
	usercount.rename(columns={'song_id': 'song_count'}, inplace=True)
	print songcount.dtypes
	print usercount.dtypes
	df_train_merged = df_train_merged.merge(songcount, on='song_id', how='left')
	df_test_merged = df_test_merged.merge(songcount, on='song_id', how='left')

	df_train_merged = df_train_merged.merge(usercount, on='msno', how='left')
	df_test_merged = df_test_merged.merge(usercount, on='msno', how='left')
	print df_train_merged.dtypes
	# he values People local and People global are new in test set. replace by NAN
	df_test_merged['source_screen_name'] = df_test_merged['source_screen_name'].replace(
		['People local', 'People global'], np.nan)

	newfeatcol = ['genre_ids_len', 'artist_name_len', 'composer_len', 'lyricist_len', 'is_featured', 'artist_composer',
				  'artist_composer_lyricist', 'msno_count', 'song_count']
	return df_train_merged[newfeatcol], df_test_merged[newfeatcol]

def preprocessing_source_type_source_screen_name_source_type():


	print('Loading data...')
	data_path = "../data/"
	train = pd.read_csv(data_path + 'train.csv')
	test = pd.read_csv(data_path + 'test.csv')
	data = pd.concat((train[['msno', 'song_id','source_type','source_screen_name','source_system_tab']],
					  test[['msno', 'song_id','source_type','source_screen_name','source_system_tab']]))
	print 'example size before drop duplicates'
	print len(data)

	data.drop_duplicates(subset=['msno', 'song_id'], keep="first", inplace=True)
	print 'example size after drop duplicates'
	print len(data)
	newfeat = None
	groups = data.groupby('msno')
	#print data
	#print  groups['source_type'].count()
	for feature in ['source_type','source_screen_name','source_system_tab']:
		valuecount = groups[feature].value_counts(normalize=True,dropna=False)
		print valuecount
		valuecount = valuecount.unstack(level=-1).add_prefix(feature)
		# countsum = valuecount.sum(axis=0)
		# for col in valuecount.columns:
		# 	valuecount[col] *= 1.0
		# 	valuecount[col] /= countsum

		if newfeat is None:
			newfeat = valuecount
		else:
			newfeat=newfeat.join(valuecount,how='left')
	newfeat.reset_index(inplace=True)
	train = train[['msno']].merge(newfeat,on='msno',how='left')
	test = test[['msno']].merge(newfeat, on='msno', how='left')
	return train,test

def computeSVD(urm, K):
	U, s, Vt = sparsesvd(urm, K)
	dim = (len(s), len(s))
	S = np.zeros(dim, dtype=np.float32)
	for i in range(0, len(s)):
		S[i,i] = mt.sqrt(s[i])
	# U = csr_matrix(np.transpose(U), dtype=np.float32)
	# S = csr_matrix(S, dtype=np.float32)
	# Vt = csr_matrix(Vt, dtype=np.float32)
	return np.transpose(U), S.dot(Vt)

def preprocessingwordembedding(latent=10):
	print('Loading data...')
	data_path = "../data/"
	train = pd.read_csv(data_path + 'train.csv')
	test = pd.read_csv(data_path + 'test.csv')
	songs = pd.read_csv(data_path + 'songs.csv')
	members = pd.read_csv(data_path + 'members.csv')
	songsinfo = pd.read_csv(data_path + "song_extra_info.csv")
	print('Data preprocessing...')

	data = pd.concat((train[['msno', 'song_id']], test[['msno', 'song_id']]))
	data = data.merge(songs, on='song_id', how='left')
	data = data.merge(members, on='msno', how='left')
	data = data.merge(song_extra_info, on='song_id', how='left')






def lgbm2(X_train, X_valid, y_train, y_valid, X_test, X_test_ids,
		 params={'learning_rate': 0.4, "application": 'binary', "max_depth": 15, 'num_leaves': 2 ** 8,
				 'verbosity': 0, 'metric': 'auc', 'num_threads': 60}, num_boost_round=9000, early_stopping_rounds=10,
		 categorical_feature=None,learning_rates=None,w=None,along=None,type=0):
	# p_test = bst.predict(X_test)
	import math
	weight = np.array(range(len(X_train)))

	if along is not None:
		day = pd.DataFrame(along)
		day.columns = ['timestamp']
		dayunique = day['timestamp'].unique()
		dayunique.sort()
		mark = pd.DataFrame(dayunique)
		mark.columns = ['timestamp']
		mark['rank'] = range(len(mark))
		day = day.merge(mark, on='timestamp')
		weight = day['rank'].values

	if w is not None:
		if type == 0:
			weight = w*np.log((weight+1))+1
		elif type == 1:
			weight = -w * np.log((-weight + 1)+len(X_train)) + w*math.log(len(X_train)+1)+1
	else:
		weight = [1] * len(X_train)
	print w
	print len(weight)
	print len(X_train)

	d_train = lgb.Dataset(X_train,weight=weight, label=y_train)
	d_valid = lgb.Dataset(X_valid, label=y_valid)

	watchlist = [d_train, d_valid]

	print('Training LGBM model...')

	model = lgb.train(params, train_set=d_train, num_boost_round=num_boost_round, valid_sets=watchlist,
					  early_stopping_rounds=early_stopping_rounds, verbose_eval=10,learning_rates=learning_rates)
	# if categorical_feature is not None:
	# 	model = lgb.train(params, train_set=d_train, num_boost_round=num_boost_round, valid_sets=watchlist,
	# 				  early_stopping_rounds=early_stopping_rounds, verbose_eval=10,categorical_feature = categorical_feature)
	print('Making predictions and saving them...')

	p_valid = model.predict(X_valid)
	p_test = model.predict(X_test)

	subm = pd.DataFrame()
	subm['id'] = X_test_ids
	subm['target'] = p_test
	# subm.to_csv('submission.csv.gz', compression='gzip', index=False, float_format='%.6f')
	print('Done!')
	print "split feature importance"
	print model.feature_importance(importance_type='split')
	print "gain feature importance"
	print model.feature_importance(importance_type='gain')

	return subm , pd.DataFrame(p_valid),model

