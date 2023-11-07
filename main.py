from DataPreparation.CryptoPreprocessor import CryptoPreprocessor
from DataPreparation.TweetPreprocessor import TweetPreprocessor
from HypothesisTesting import HypothesisTesting
from sklearn.metrics import balanced_accuracy_score
from gensim.test.utils import datapath
from gensim.models import LdaMulticore
from xgboost import XGBClassifier
from bertopic import BERTopic 
from datetime import datetime
from textwrap import dedent
import pandas as pd
import DashApp 
import json
import sys


def get_user_daterange(args):
    format_date = lambda x: pd.Timestamp(datetime.strptime(x, '%Y-%m-%d'), tz='UTC')
    try:
        user_min = format_date(args[1])
        user_max = format_date(args[2])
    except:
        while True:
            ans = input('Do you want to specify date range? (Y/N)\n').lower()
            if ans=='y':
                print('--Type dates in yyyy-mm-dd format--')
                start = str(input("Enter start date: "))
                end = str(input("Enter end date: "))
                user_min = format_date(start)
                user_max = format_date(end)
                break
            elif ans=='n':
                user_min = format_date('2002-01-01')
                user_max = format_date('3002-01-01')
                break                
            else:
                print("Invalid input.")
    
    return user_min, user_max

def set_data_range(user_min, user_max, tweets, btc_df):
    tweets = tweets.copy()
    btc_df = btc_df.copy()
    
    MIN = tweets['date'].iloc[0]
    MAX = tweets['date'].iloc[-1]    
    if (user_min >= MIN) & (user_max <= MAX):   
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
    tweets = pd.read_csv(r'Data/elon_tweets.csv', index_col=0)
    btc_df = pd.read_csv('Data/btc_data.csv', index_col=0)    
    tweets['date'] = pd.to_datetime(tweets['date'])  
    btc_df['time'] = pd.to_datetime(btc_df['time'])
    
    user_min, user_max = get_user_daterange(args)
    tweets, btc_df = set_data_range(user_min, user_max, tweets, btc_df)
    mod_tweets_df = TweetPreprocessor(tweets).transform()
   
    bertopic_model = BERTopic.load("Models/NLPmodels/BERTopic_v1")
    fig = bertopic_model.visualize_hierarchy()
    fig.write_html("hierarchical_topics_v1.html")
    
    crypto_prep = CryptoPreprocessor()
    new_topics_btc = crypto_prep.transform(bertopic_model, mod_tweets_df, btc_df)
    new_topics_btc, new_predictors = crypto_prep.add_trend_season(data=new_topics_btc, 
                                                                  horizons=[2,7,21,28,60,90,180,364])
    xgb_model = XGBClassifier()
    xgb_model.load_model('Models/CRYPTOmodels/xgbc_bertopic.json')
    filtered_data = new_topics_btc[xgb_model.feature_names_in_].copy()
    predictions = xgb_model.predict(filtered_data)
    
    return predictions, new_topics_btc
    
    
if __name__ == '__main__': 
    show_dash_app = True if sys.argv[-1].lower() == 'true' else False
    
    predictions, btc_df = main(sys.argv)
    conclusion_str = f"""\
    Predicted:
        up:   {len(predictions[predictions==1])}
        down: {len(predictions[predictions==0])}
    Actual:
        up:   {len(btc_df[btc_df['target']==1])}
        down: {len(btc_df[btc_df['target']==0])}
        
    Accuracy: {balanced_accuracy_score(btc_df['target'], predictions):.3f}"""
    
    ttest_str, wicoxon_str = HypothesisTesting(alpha=0.05).get_results()
    print(dedent(conclusion_str))
    print('------------------\n')
    
    print('Hypothesis testing')
    print(dedent(ttest_str))
    print(dedent(wicoxon_str))
    
    if show_dash_app==True:
        DashApp.run_app(debug=True)