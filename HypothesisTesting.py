from scipy import stats
import pandas as pd
import numpy as np


class HypothesisTesting:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.param_grid = {'n_estimators':np.arange(5,30,1),
                           'max_depth':np.arange(2,10,1),
                           'gamma':np.linspace(0, 5, 10),
                           'alpha':np.arange(1,50,10),
                           'lambda':np.arange(1,50,10),
                           'learning_rate':np.arange(0.01, 0.5, 0.01),
                           'max_delta_step':np.linspace(0,1,10),
                           'grow_policy':['depthwise', 'lossguide']
                           }
        
        self.A_group = pd.concat([pd.read_csv('Models/Tuned_in_csv/with_tweets_models (1).csv', index_col=0), 
                                  pd.read_csv('Models/Tuned_in_csv/with_tweets_models.csv', index_col=0)])
        
        self.B_group = pd.concat([pd.read_csv('Models/Tuned_in_csv/without_tweets_models (1).csv', index_col=0), 
                                  pd.read_csv('Models/Tuned_in_csv/without_tweets_models.csv', index_col=0)])
    
    def ttest(self, df):
        tstat, pvalue = stats.ttest_ind(df['with_tweets_scores'], df['without_tweets_scores'], equal_var=False)
        return pvalue < self.alpha
        
    def wilcoxon(self, df):
        tstat, pvalue = stats.wilcoxon(df['with_tweets_scores'], df['without_tweets_scores'])
        return pvalue < self.alpha
    
    def load_data(self):
        df = pd.concat([self.A_group['mean_test_score'], 
                        self.B_group['mean_test_score']],
                       axis=1, verify_integrity=True)
        
        df.columns = ['with_tweets_scores','without_tweets_scores']
        
        return df
    
    def get_results(self):      
        """Returns result from ttest and wilcoxon test saying whether there's a relationship or not

        Returns:
            ttest_str, wicoxon_str
        """        
        df = self.load_data()
        
        ttest_str = ""
        if self.ttest(df) == True:
            ttest_str = f'''\
            By ttest:
                The pvalue is less than alpha={self.alpha}, hypothesis should be rejected due to low 
                probability of this to be true suggesting that there's no relationship between Elon Musk and cryptocurrency.'''
        else:
            ttest_str = f'''\
                The pvalue is greater than alpha={self.alpha}, 
                there's likely a relationship between Elon Musk's tweets and cryptocurrency.'''
            
        wicoxon_str = ""
        if self.wilcoxon(df) == True:
            wicoxon_str = f'''\
            By Wilcoxon test:
                The pvalue is less than alpha={self.alpha}, hypothesis should be rejected due to low 
                probability of this to be true suggesting that there's no relationship between Elon Musk and cryptocurrency.'''
        else:
            wicoxon_str = f'''\
                The pvalue is greater than alpha={self.alpha}, 
                there's likely a relationship between Elon Musk's tweets and cryptocurrency.'''
            
        return ttest_str, wicoxon_str

   