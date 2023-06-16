from gensim.models import LdaMulticore
from gensim.test.utils import datapath
from TopicModeling import TopicModeling


class ModelRepoNLP(TopicModeling):
    '''
    Saves LDA gensim model to database which contains few previous models for a backup.
    '''
    mother_path = datapath(r"D:\Projects\ElonMuskCrypto\Models\NLPmodels")
    
    def __init__(self, lda_model=None):
        self.lda_model = lda_model
        self.path = datapath(f"{self.mother_path}\lda")
    
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