from .TextVectorizer import TextVectorizer
from datetime import datetime
import pandas as pd
from . import utils


class CryptoPreprocessor: 
    def __insert_topics(self, lda_model, tweets_df, topics_preffix='T_', topics_suffix=''):        
        preprocessing_pipeline = TextVectorizer().make_pipeline()
        tweets_df['vectorized']  = preprocessing_pipeline.transform(tweets_df['rawContent'].values.tolist())[1]
        tweets_df['TopicsProbs'] = tweets_df['vectorized'].apply(lambda x: dict(lda_model.get_document_topics(x, minimum_probability=0)))
        
        topics_df = pd.DataFrame(tweets_df['TopicsProbs'].tolist())
        for col in topics_df.columns:
            rename_pattern = f'{topics_preffix}{col}{topics_suffix}'
            topics_df = topics_df.rename({col: rename_pattern}, axis=1)

        topics_tweets_df = pd.concat([tweets_df, topics_df], axis=1).copy()
        topics_tweets_df = topics_tweets_df.drop(['vectorized', 'TopicsProbs'], axis=1)
            
        return topics_tweets_df
            
    def __merge_data(self, lda_model, tweets_df, raw_crypto_df):
        '''Merges tweets with discovered topics via LDA model and then with raw crypto df for further Time-series analysis.'''
        topics_tweets_df = self.__insert_topics(lda_model, tweets_df).drop('rawContent', axis=1)
        topics_tweets_df['date'] = pd.to_datetime(topics_tweets_df['date']).dt.date
        
        aggs = utils.make_aggregator(topics_tweets_df)
        grouped_df = (topics_tweets_df
                      .groupby('date')
                      .agg(aggs).reset_index())

        raw_crypto_df = raw_crypto_df.rename(columns={'time':'date'}).drop('conversionType', axis=1)
        raw_crypto_df['date'] = pd.to_datetime(raw_crypto_df['date']).dt.date
        
        # TODO: change how to left
        crypto_topics = pd.merge(raw_crypto_df, grouped_df, on='date', how='left')
        
        if 'conversionSymbol' in crypto_topics.columns:
            crypto_topics = crypto_topics.drop('conversionSymbol', axis=1)
            
        return crypto_topics
    
    def create_date_features(self, df):
        df = df.set_index('date')
        df.index = pd.to_datetime(df.index)
        
        df['day'] = df.index.day
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        
        return df
    
    def transform(self, lda_model, tweets_df, raw_crypto_df):
        df = self.__merge_data(lda_model, tweets_df, raw_crypto_df)
        df = self.create_date_features(df)
        
        df["tomorrow"] = df["close"].shift(-1)
        df["target"] = (df["tomorrow"] > df["close"]).astype(int)
        df = df.reset_index()
        df['isPandemic'] = df['date'].apply(lambda x: 1 if (datetime(2020,1,30) <= x) & (x <= datetime(2023,5,5)) else 0)
        df = df.set_index('date')
        df = self.add_trend_season(df, [2,7,14,50,90])
        
        return df
    
    def add_trend_season(self, data, horizons, ignore_trend=False, ignore_season=False):
        data = data.copy()
        predictors = []
        for horizon in horizons:
            rolling_average = data.rolling(horizon).mean()
            
            if not ignore_trend:
                ratio_column = f'close_ratio{horizon}'
                data[ratio_column] = data['close'] / rolling_average['close']
                predictors += [ratio_column]
                
            if not ignore_season:            
                trend_column = f'Trend_{horizon}'
                data[trend_column] = data.shift(1).rolling(horizon).sum()['target']
                predictors += [trend_column]        
            
        return data