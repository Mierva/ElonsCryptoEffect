from gensim.models import LdaMulticore
from gensim.test.utils import datapath


class ModelRepoCrypto:
    '''
    
    '''
    models_path = r"D:\Projects\ElonMuskCrypto\CRYPTOmodel"
    
    def __init__(self, lda_model=None, model_name='crypto_model'):
        self.lda_model = lda_model
        self.model_name = model_name
        self.path = datapath(f"D:\Projects\ElonMuskCrypto\CRYPTOmodel\{model_name}")
    
    def __connect_to_db(self):
        #TODO: well, yea
        pass
    
    def save_model(self):
        #TODO: save to db
        self.lda_model.save(f'{self.path}\{self.model_name}')
        
    def load_model(self):
        #TODO: load to db
        model = LdaMulticore.load(self.path)
        return model