from DataPreparation.CryptoPreprocessor import CryptoPreprocessor
from DataPreparation.TweetPreprocessor import TweetPreprocessor
from DataPreparation.TextVectorizer import TextVectorizer
from datetime import datetime, timedelta
from gensim.test.utils import datapath
from gensim.models import LdaMulticore
from TweetScraper import TweetScraper
from pyLDAvis.gensim import prepare 
from xgboost import XGBClassifier
from CryptoApi import CryptoApi
import pandas as pd
import numpy as np
import pyLDAvis
import json
import sys
import re


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

def main(args):
    # try:
    #     start = args[1] 
    #     end = args[2]  
    # except:
    #     start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d") 
    #     end = datetime.now().strftime("%Y-%m-%d")
    
    # scrp = TweetScraper(start=start, end=ewnd, max_empty_pages=1, max_workers=8)
    # new_tweets = scrp.parallel_download_tweets()
                    
    user_min, user_max = get_user_daterange(args)
    
    tweets = pd.read_csv(r'Data/elon_tweets.csv', index_col=0)
    btc_df = pd.read_csv('Data/btc_data.csv')    
    tweets['date'] = pd.to_datetime(tweets['date'])  
    btc_df['time'] = pd.to_datetime(btc_df['time'])
    
    MIN = tweets['date'].iloc[0]
    MAX = tweets['date'].iloc[-1]
    if (user_min >= MIN) & (user_max <= MAX):
        # start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d") 
        # end = datetime.now().strftime("%Y-%m-%d")
        
        # scrp = TweetScraper(start=args[1], end=args[2], max_empty_pages=1, max_workers=8)
        # new_tweets = scrp.parallel_download_tweets()     
        tweet_mask = (tweets['date'] >= user_min) & (tweets['date'] <= user_max)
        tweets = tweets[tweet_mask]
        btc_mask   = (btc_df['time'].dt.date >= user_min.date()) & (btc_df['time'].dt.date <= user_max.date())
        btc_df = btc_df[btc_mask]
    
    # with open('crypto_token.txt','r') as f:
    #     token = f.readline()

    # crypto = CryptoApi(token)
    # period_count = (datetime.strptime(end,'%Y-%m-%d')-datetime.strptime(start,'%Y-%m-%d')).days    
    # new_btc = crypto.fetch_data('btc','usd', period='day', period_count=period_count)

    # new_tweets = pd.DataFrame(new_tweets)
    tweets_df = pd.read_csv(r'Data/elon_tweets.csv', index_col=0)
    allowed_cols = [col for col in tweets.columns if col in tweets_df.columns]
    tweets = tweets[allowed_cols].copy()

    twt_prep = TweetPreprocessor(tweets)
    mod_tweets_df = twt_prep.transform()
    text2vec = TextVectorizer()
    preprocessing_pipeline = text2vec.make_pipeline()
    id2word, corpus = preprocessing_pipeline.transform(mod_tweets_df['rawContent'].values.tolist())

    temp_file = datapath(r"D:\Projects\ElonMuskCrypto\Models\NLPmodels\lda")
    lda_model = LdaMulticore.load(temp_file)
    
    p = prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(p, 'lda.html')
    
    crypto_prep = CryptoPreprocessor()
    new_topics_btc = crypto_prep.transform(lda_model, mod_tweets_df, btc_df)
    horizons = [2,7,21,28,60,90,180,364] 
    new_topics_btc, new_predictors = crypto_prep.add_trend_season(new_topics_btc, horizons)
    predictors = new_topics_btc.columns[new_topics_btc.columns!='target']
    
    xgb_model = XGBClassifier()
    xgb_model.load_model('Models/CRYPTOmodels/xgbc_1694187252.json')

    allowed_predictors = new_topics_btc[xgb_model.feature_names_in_].copy()
    predictions = xgb_model.predict(allowed_predictors)
    
    return predictions
    
if __name__ == '__main__':
    predictions = main(sys.argv)
    print(f'preds: {predictions.tolist()}')
    