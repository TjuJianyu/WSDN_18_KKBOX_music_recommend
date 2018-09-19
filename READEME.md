# WSDN_18_KKBOX_music_recommend


## description 
That is the code for 6th solution at Kaggle KKBox music recommendation challenge [here](https://www.kaggle.com/c/kkbox-music-recommendation-challenge/discussion/45999).
Also, that is the code corrsponding to paper "KKbox’s Music Recommendation Challenge Solution
with Feature engineering" [here](http://wsdm-cup-2018.kkbox.events/pdf/WSDM_KKboxs_Music_Recommendation_Challenge_6th_Solution.pdf).

Detail are shown in "KKbox’s Music Recommendation Challenge Solution
with Feature engineering". Ensamble part is not included here.


### user\_song\_simi\_contextbase.py & user\_song\_simi\_songbase.py
That is the code for calculating similarity of user and song by different features. This part uses parallel from Joblib to speed up.

### user\_artist\_svd.py
That is the code for SVD represenation of user and artist by sparsesvd at [here](https://pypi.org/project/sparsesvd/).

### prepro\_and\_train.py
That is the code of other basic preprocessing, feature engineering and classification model.

### train.py
Training and dumping the model and results in ../out/


## Workflow
1. Download dataset from [here](https://www.kaggle.com/c/kkbox-music-recommendation-challenge/) and put it in /data/
2. run 

	user\_song\_simi\_contextbase.py 
	
	user\_song\_simi\_songbase.py
	
	user\_artist\_svd.py
	
	to generate similarity and SVD features
	
3. run train.py to generate other features and train a classification model.


	