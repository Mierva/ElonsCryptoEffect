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
        '''Merges tweets with discovered topics via LDA model and then with raw crypto df.'''
        
        topics_tweets_df = self.__insert_topics(lda_model, tweets_df).drop('rawContent', axis=1)
        topics_tweets_df['date'] = pd.to_datetime(topics_tweets_df['date']).dt.date
        
        aggs = utils.make_aggregator(topics_tweets_df)
        grouped_df = (topics_tweets_df
                      .groupby('date')
                      .agg(aggs).reset_index())

        raw_crypto_df = raw_crypto_df.rename(columns={'time':'date'}).drop('conversionType', axis=1)
        raw_crypto_df['date'] = pd.to_datetime(raw_crypto_df['date']).dt.date
        
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
        
        df['is_month_end'] = df.index.is_month_end==True
        df['is_month_start'] = df.index.is_month_start==True
        df['is_quarter_end'] = df.index.is_quarter_end==True
        df['is_quarter_start'] = df.index.is_quarter_start==True
        df['is_year_end'] = df.index.is_year_end==True
        df['is_year_start'] = df.index.is_year_start==True
        
        return df
    
    def add_lags(self,df):
        df = df.copy()
        target_map = df['target'].to_dict()
        df['lag1'] = (df.index - pd.Timedelta('1 day')).map(target_map) 
        df['lag7'] = (df.index - pd.Timedelta('7 day')).map(target_map) 
        df['lag14'] = (df.index - pd.Timedelta('14 day')).map(target_map) 
        df['lag30'] = (df.index - pd.Timedelta('30 day')).map(target_map) 
        df['lag60'] = (df.index - pd.Timedelta('60 day')).map(target_map) 
        df['lag90'] = (df.index - pd.Timedelta('90 day')).map(target_map) 
        df['lag365'] = (df.index - pd.Timedelta('365 day')).map(target_map) 
        return df
    
    def transform(self, lda_model, tweets_df, raw_crypto_df):
        df = self.__merge_data(lda_model, tweets_df, raw_crypto_df)
        df = self.create_date_features(df)
        
        # df["tomorrow"] = df["close"].shift(-1)
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df = self.add_lags(df)
        # df = df.reset_index()
        # df['isPandemic'] = df['date'].apply(lambda x: 1 if (datetime(2020,1,30) <= x) & (x <= datetime(2023,5,5)) else 0)
        # df = df.set_index('date')
        
        return df
    
    def add_trend_season(self, data, horizons, ignore_trend=False, ignore_season=False):
        data = data.copy()
        predictors = []
        for horizon in horizons:
            rolling_average = data['close'].rolling(horizon).mean()
            
            if not ignore_trend:
                rolling = f'rolling{horizon}'
                data[rolling] = rolling_average
                predictors += [rolling]
                
            if not ignore_season:            
                trend_column = f'Trend_{horizon}'
                # data[f'{trend_column}'] = data.shift(1).rolling(horizon).sum()['target']
                data[f'{trend_column}'] = data.shift(1).rolling(horizon).mean()['target']
                predictors += [trend_column]        
            
        return data, predictors