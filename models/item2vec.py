import os
try:
    import cPickle
except ImportError:
    import pickle as cPickle

import fire
import tqdm

from util import iterate_data_files

import gensim
from gensim.models import Word2Vec

from models.mostpopular import MostPopular 


# python train.py --from-dtm 0 --to-dtm 2019030100 recommend ./res/predict/dev.users ./submission/item2vec_recommend.txt

class Item2Vec(object):
    topn = 100

    def __init__(self, from_dtm, to_dtm, tmp_dir='./tmp/'):
        self.from_dtm = str(from_dtm)
        self.to_dtm = str(to_dtm)
        self.tmp_dir = tmp_dir 
        self.mp = MostPopular(from_dtm, to_dtm, tmp_dir)

    def _get_model_path(self):
        model_path = os.path.join(self.tmp_dir, 'item2vec.model.%s.%s' % (self.from_dtm, self.to_dtm))
        return model_path
    
    def _get_model(self):
        model_path = self._get_model_path()
        model = Word2Vec.load(model_path)

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
        print(seens)
        return seens

    def _make_positive_list(self, model, positive_list):
        arr = []
        for positive in positive_list:
            if positive in model.wv.vocab.keys():
                arr.append(positive)
        return arr
    

    def _recommender(self, model, positive_list=None, negative_list=None, topn=20):
        recommend_movie_ls = []
        
        for article_id, prob in model.wv.most_similar_cosmul(positive=positive_list, negative=negative_list, topn=topn):
            recommend_movie_ls.append(article_id)
        return recommend_movie_ls

    def recommend(self, userlist_path, out_path):
        model = self._get_model()
        

        with open(out_path, 'w') as fout:
            users = [u.strip() for u in open(userlist_path)]
            seens = self._get_seens(users) # positive list
            recommend_list = self.mp.get_recommend_list(seens, topn=self.topn) # user cold strat 어떻게 해결하지? => 일단 most popular에서 가져옴
            for user in users:
                seen = list(set(seens.get(user, [])))
                positive_list = self._make_positive_list(model, seen)
                if len(positive_list) > 0: 
                    recommend_list = self._recommender(model, positive_list=positive_list, topn=self.topn)
                fout.write('%s %s\n' % (user, ' '.join(recommend_list)))