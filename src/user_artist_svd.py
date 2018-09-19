
from  tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sparsesvd import sparsesvd
import joblib
import numpy as np
import math
import pandas as pd

data_path = "../data/"
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
songs = pd.read_csv(data_path + 'songs.csv')
members = pd.read_csv(data_path + 'members.csv')
data = pd.concat((train[['msno','song_id','target']],test[['msno','song_id']]))
data = data.drop_duplicates()
songs = pd.read_csv(data_path + 'songs.csv')


for col in ['artist_name' ]:#,'composer','lyricist','genre_ids']:
    songs[col] = songs[col].str.replace(r'\([\s\S]+?\)','')
    songs[col] = songs[col].str.replace(r'（[\s\S]+?）','')
    songs[col].fillna('',inplace=True)
data = data.merge(songs,on='song_id',how='left')
data['artist_name'].fillna('',inplace=True)
user_groups = data.groupby('msno')
msnoindex = user_groups['artist_name'].count().reset_index(drop=False)
msnoindex['msno_id'] = range(len(msnoindex))
data = data.merge(msnoindex[['msno','msno_id']],on='msno',how='left')

newfeat = []
matrixlist = []
lelist = []
#for col in ['artist_name']:#,'composer','lyricist','genre_ids']:
col = 'artist_name'
feat = user_groups[col].agg(lambda x: "|".join(x))
userfeat = feat.str.split(r',|\||\\|&|\+|;|、| ，|and|/|feat.|features|featuring|with| X |')
index=[]
value=[]
farray=userfeat
for i in tqdm(range(len(farray))):
    vals = farray[i]
    if type(vals) is not list:
        pass
    else:
        for val in vals:
            value.append(val.strip())
            index.append(i)

le = LabelEncoder()
le.fit(value)
label = le.transform(value)
matrix = csr_matrix(([1]*len(label),(index,label)),shape=(len(farray),max(label)+1))
matrixlist.append(matrix)
lelist.append(le)

def computeSVD(urm, K):
    U, s, Vt = sparsesvd(urm, K)
    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i,i] = math.sqrt(s[i])
    return np.transpose(U), S.dot(Vt)


index=[]
value=[]
farray=data[col].str.split(r',|\||\\|&|\+|;|、| ，|and|/|feat.|features|featuring|with| X |').values
for i in tqdm(range(len(farray))):
    vals = farray[i]
    if type(vals) is not list:
        pass
    else:
        for val in vals:
            value.append(val.strip())
            index.append(i)
itemfeat = le.transform(value)

lastone = index[0]
featstack = []
inter= []
for i in tqdm(xrange(len(itemfeat))):
    if index[i] == lastone:
        inter.append(itemfeat[i])
        if i == len(index)-1:
            featstack.append(inter)
    else:
        featstack.append(inter)
        inter = []
        inter.append(itemfeat[i])
        lastone = index[i]

msno_id = data['msno_id'].values

svdsimi = None
for latentspace in [180]:
    U, V = computeSVD(matrix.tocsc(), latentspace)
    similist = []
    simimax = []
    for i in tqdm(xrange(len(featstack))):
        simi_iter = []
        simi_max = 0
        for val in featstack[i]:

            simi = U[msno_id[i]].dot(V[:,val])
            simi_iter.append(simi)
            if simi>simi_max:
                simi_max = simi
#             if simi<simi_min:
#                 simi_min = simi
#             simi_mean += simi
#         if len(featstack[i]) >0:
#             simi_mean /=len(featstack[i])
#         similist.append(simi_iter)
        simimax.append(simi_max)
    svdsimi = pd.Series(simimax)
    print pd.Series(simimax[:len(train)]).corr(train['target'])

pd.DataFrame(svdsimi).to_csv("../out/simi/artist_name_svddot.csv",index=None,header=None)
