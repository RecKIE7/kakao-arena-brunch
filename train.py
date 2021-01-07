import fire
from models.mostpopular import MostPopular 

class Train(object):
    def __init__(self, from_dtm, to_dtm, tmp_dir='./tmp/'):
        self.model = MostPopular(from_dtm, to_dtm, tmp_dir)

    def recommend(self, userlist_path, out_path):
        self.model.recommend(userlist_path, out_path)
    
    def train(self):
        pass

if __name__ == '__main__':
    fire.Fire(Train)