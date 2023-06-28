from DataPreparation.CryptoPreprocessor import CryptoPreprocessor
from DataPreparation.TweetPreprocessor import TweetPreprocessor
from DataPreparation.TextVectorizer import TextVectorizer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import precision_score
from gensim.test.utils import datapath
from gensim.models import LdaMulticore
from TweetScraper import TweetScraper
from xgboost import XGBClassifier
from CryptoApi import CryptoApi
from datetime import datetime
import pandas as pd
import numpy as np
import re


def main():
    # values set by a user
    start = '2023-06-10'
    end = '2023-06-20'
    scrp = TweetScraper(start=start, end=end, max_empty_pages=1, max_workers=8)
    new_tweets = scrp.parallel_download_tweets()

    with open('crypto_token.txt','r') as f:
        token = f.readline()

    crypto = CryptoApi(token)
    period_count = (datetime.strptime(end,'%Y-%m-%d')-datetime.strptime(start,'%Y-%m-%d')).days    
    new_btc = crypto.fetch_data('btc','usd', period='day', period_count=period_count)

    new_tweets = pd.DataFrame(new_tweets)
    tweets_df = pd.read_csv(r'Data/elon_tweets.csv', index_col=0)
    allowed_cols = [col for col in new_tweets.columns if col in tweets_df.columns]
    new_tweets = new_tweets[allowed_cols].copy()

    twt_prep = TweetPreprocessor(new_tweets)
    mod_tweets_df = twt_prep.transform()
    text2vec = TextVectorizer()
    preprocessing_pipeline = text2vec.make_pipeline()
    id2word, corpus = preprocessing_pipeline.transform(mod_tweets_df['rawContent'].values.tolist())

    temp_file = datapath(r"D:\Projects\ElonMuskCrypto\Models\NLPmodels\lda")
    lda_model = LdaMulticore.load(temp_file)
    crypto_prep = CryptoPreprocessor()
    new_topics_btc = crypto_prep.transform(lda_model, mod_tweets_df, new_btc)
    
    xgb_model = XGBClassifier()
    xgb_model.load_model('Models/CRYPTOmodels/xgb_6226415094339622.json')

    allowed_predictors = new_topics_btc[xgb_model.feature_names_in_].copy()
    predictions = xgb_model.predict(allowed_predictors)

    return predictions
    
if __name__ == '__main__':
    predictions = main()
    print(f'preds: {predictions}')
    