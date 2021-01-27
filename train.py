import fire
from models.mostpopular import MostPopular 
# from models.item2vec import Item2Vec 
from models.knn import KNN 

class Train(object):
    def __init__(self, from_dtm, to_dtm):
        self.model = KNN(from_dtm, to_dtm)


    def recommend(self, userlist_path, out_path):
        self.model.recommend(userlist_path, out_path)
    
    def train(self):
        pass

if __name__ == '__main__':
    fire.Fire(Train)