import os, sys
try:
    import cPickle
except ImportError:
    import pickle as cPickle

import fire
import tqdm
from tqdm.contrib import tzip

from util import iterate_data_files

import gensim
from gensim.models import Word2Vec

from models.mostpopular import MostPopular 

import pickle 
import joblib
import numpy as np
sys.path.append('..')
sys.path.append('../../')
import config as conf
# python train.py --from-dtm 0 --to-dtm 2019030100 recommend ./res/predict/dev.users ./submission/knn_50.txt

class KNN(object):
    topn = 100

    def __init__(self, from_dtm, to_dtm, chk_dir='./checkpoints/', tmp_dir='./tmp/'):
        self.max_len = 70
        self.from_dtm = str(from_dtm)
        self.to_dtm = str(to_dtm)
        self.tmp_dir = tmp_dir 
        self.chk_dir = chk_dir
        self.mp = MostPopular(from_dtm, to_dtm, tmp_dir)
        self.dictionary = self._get_dictionary()
        

    def _get_model_path(self):
        model_path = os.path.join(self.chk_dir, 'knn_'+str(self.max_len)+'.sav')
        return model_path


    def _get_model(self):
        model_path = self._get_model_path()
        model = joblib.load(model_path) 

        return model

    def _get_seens(self, users):
        set_users = set(users)
        seens = {}
        for path, _ in tqdm.tqdm(iterate_data_files(self.from_dtm, self.to_dtm),
                                 mininterval=1):
            for line in open(path):
                tkns = line.strip().split()
                userid, seen = tkns[0], tkns[1:]
                if userid not in set_users:
                    continue
                seens[userid] = seen
        # print(seens)
        return seens
    
    def _get_dictionary(self):
        print(conf.root+'res/dictionary_'+str(self.max_len)+'.pickle')
        with open(conf.root+'res/dictionary_'+str(self.max_len)+'.pickle', 'rb') as fr:
            dictionary = pickle.load(fr)
            return dict(dictionary)

    
    def _get_articles(self, users):
        test_X = np.load(open(conf.root + 'res/dev_users_'+str(self.max_len)+'.npy', 'rb'))
        return test_X

    # ./submission/knn_50.txt
    def recommend(self, userlist_path, out_path):
        model = self._get_model()
        user_list = np.load(open(conf.root + 'res/user_list_'+str(self.max_len)+'.npy', 'rb'))
        with open(out_path, 'w') as fout:
            users = [u.strip() for u in open(userlist_path)]
            seens = self._get_seens(users) # positive list
            test_X = self._get_articles(users)
            mp_list = self.mp.get_recommend_list(seens, topn=self.topn) # user cold strat 어떻게 해결하지? => 일단 most popular에서 가져옴            
       
            for user, articles in tzip(users, test_X):
                recommend = []
                left = []
                if len(articles) == 0: # 기존에 읽은 히스토리가 없으면. (MP)
                    recommend = mp_list
                else: # 기존에 읽은 히스토리가 있으면. (KNN)
                    pred = model.kneighbors([articles])
                    sim_users = pred[1][0]
                    dist = pred[0][0]
                    for i, u in enumerate(sim_users):
                        if dist[i] != 0:
                            recommend += self.dictionary[user_list[u]]

                if user in self.dictionary.keys():
                    recommend = list(set(recommend) - set(self.dictionary[user]))
                else:
                    recommend = list(set(recommend))
                    
        #         np.random.shuffle(recommend)

                if len(recommend) < 100:
                    recommend += mp_list
                recommend = list(set(recommend))

                fout.write('%s %s\n' % (user, ' '.join(recommend[:100])))
