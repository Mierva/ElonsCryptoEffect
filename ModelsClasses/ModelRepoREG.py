from gensim.models import LdaMulticore
from gensim.test.utils import datapath


class ModelRepoREG:
    '''
    
    '''
    models_path = r"D:\Projects\ElonMuskCrypto\REGmodel"
    
    def __init__(self, lda_model=None):
        self.lda_model = lda_model
        self.path = datapath(r"D:\Projects\ElonMuskCrypto\REGmodel\lda")
    
    def __connect_to_db(self):
        #TODO: well, yea
        pass
    
    def save_model(self):
        #TODO: save to db
        self.lda_model.save(f'{self.path}\lda')
        
    def load_model(self):
        #TODO: load to db
        lda_model = LdaMulticore.load(self.path)
        return lda_model