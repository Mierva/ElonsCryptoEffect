from DataPreparation.CryptoPreprocessor import CryptoPreprocessor
from DataPreparation.TweetPreprocessor import TweetPreprocessor
# from DataPreparation.TextVectorizer import TextVectorizer
from sklearn.metrics import balanced_accuracy_score
from datetime import datetime
from gensim.test.utils import datapath
from gensim.models import LdaMulticore
# from gensim.corpora import Dictionary
# from TweetScraper import TweetScraper
from pyLDAvis.gensim import prepare
from xgboost import XGBClassifier
# from CryptoApi import CryptoApi
from textwrap import dedent
import pandas as pd
# import numpy as np
import pyLDAvis
import json
import sys
# import re


def get_user_daterange(args):
    FORMAT = '%Y-%m-%d'
    TZ = 'UTC'

    try:
        user_min = pd.Timestamp(datetime.strptime(args[1], FORMAT), tz=TZ)
        user_max = pd.Timestamp(datetime.strptime(args[2], FORMAT), tz=TZ)
    except:
        while True:
            ans = input('Do you want to specify date range? (Y/N)\n').lower()
            
            if ans=='y':
                print('--Type dates in yyyy-mm-dd format--')
                start = str(input("Enter start date: "))
                end = str(input("Enter end date: "))
                user_min = pd.Timestamp(datetime.strptime(start, FORMAT), tz=TZ)
                user_max = pd.Timestamp(datetime.strptime(end, FORMAT), tz=TZ)
                break
            elif ans=='n':
                user_min = pd.Timestamp(datetime.strptime(start, FORMAT), tz=TZ)
                user_max = pd.Timestamp(datetime.strptime(end, FORMAT), tz=TZ)
                break                
            else:
                print("Invalid answer.")
    
    return user_min, user_max

def set_data_range(user_min, user_max, tweets, btc_df):
    tweets = tweets.copy()
    btc_df = btc_df.copy()
    
    MIN = tweets['date'].iloc[0]
    MAX = tweets['date'].iloc[-1]    
    if (user_min >= MIN) & (user_max <= MAX):
        # start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d") 
        # end = datetime.now().strftime("%Y-%m-%d")
        
        # scrp = TweetScraper(start=args[1], end=args[2], max_empty_pages=1, max_workers=8)
        # new_tweets = scrp.parallel_download_tweets()     
        date_mask = (tweets['date'] >= user_min) & (tweets['date'] <= user_max)
        tweets = tweets[date_mask]
        btc_mask   = (btc_df['time'].dt.date >= user_min.date()) & (btc_df['time'].dt.date <= user_max.date())
        btc_df = btc_df[btc_mask]
        
    return tweets, btc_df

def load_lda_files():
    temp_file = datapath(r"D:\Projects\ElonMuskCrypto\Models\NLPmodels\lda")    
    lda_model = LdaMulticore.load(temp_file)
    id2word = lda_model.id2word
    with open('corpus.json', 'r') as r:
        corpus = json.load(r)
    
    return lda_model, id2word, corpus

def main(args):                    
    user_min, user_max = get_user_daterange(args)
    
    tweets = pd.read_csv(r'Data/elon_tweets.csv', index_col=0)
    btc_df = pd.read_csv('Data/btc_data.csv')    
    tweets['date'] = pd.to_datetime(tweets['date'])  
    btc_df['time'] = pd.to_datetime(btc_df['time'])
    
    tweets, btc_df = set_data_range(user_min, user_max, tweets, btc_df)
    mod_tweets_df = TweetPreprocessor(tweets).transform()
 
    lda_model, id2word, corpus = load_lda_files()
    pyLDAvis.save_html(prepare(lda_model, corpus, id2word), 'lda.html')
    
    crypto_prep = CryptoPreprocessor()
    new_topics_btc = crypto_prep.transform(lda_model, mod_tweets_df, btc_df)
    horizons = [2,7,21,28,60,90,180,364] 
    new_topics_btc, new_predictors = crypto_prep.add_trend_season(new_topics_btc, horizons)
    
    xgb_model = XGBClassifier()
    xgb_model.load_model('Models/CRYPTOmodels/xgbc_1694187252.json')

    allowed_predictors = new_topics_btc[xgb_model.feature_names_in_].copy()
    predictions = xgb_model.predict(allowed_predictors)
    
    return predictions, new_topics_btc
    
    
if __name__ == '__main__':
    predictions, btc_df = main(sys.argv)
    
    conclusion_str = f"""\
    Predicted:
        up: {predictions[predictions==1].shape[0]}
        down: {predictions[predictions==0].shape[0]}
    Actual:
        up: {btc_df[btc_df['target']==1].shape[0]}
        down: {btc_df[btc_df['target']==0].shape[0]}
    Accuracy: {balanced_accuracy_score(btc_df['target'], predictions):.3f}"""
    
    print(dedent(conclusion_str))
    