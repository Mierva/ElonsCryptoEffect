from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
import datetime
import json
import re

class TweetCleaner:
    dictionary = None
    
    def __init__(self, tweets):
        if type(tweets)!=pd.DataFrame:
            self.tweets_df = pd.DataFrame(tweets)
        else:
            self.tweets_df = tweets          

    def extract_dict(self, line: str, prepare_to_df=False):
        """Extracts data from a dict represented as string and makes it a dict.

        ## Parameters:
            line (str): row of a Series/DataFrame to be preprocessed.
            prepare_to_df (bool): prepares extracted dict to be wrapped into DataFrame.

        ## Returns:
            dict: extracted dict from string.
        """    

        extracted_content = dict(re.findall(r"'(\w+)': '?({.*}|datetime.datetime\(.*\)|[\w\d/:\. ]*)'?", line))
        
        # Wraps dict values into lists to be easily represented as a DataFrame row.
        if prepare_to_df:
            for key,value in extracted_content.items():
                if value == '':
                    extracted_content[key] = [None]
                else:
                    extracted_content[key] = value
            
        return extracted_content

    def clean_text(self, raw_text):    
        cleaned_text = re.sub(r' \'?(displayname|renderedDescription)\'?: (.*?)(\'|None),', '', raw_text)
        cleaned_text = (cleaned_text
                        .replace("'",'"')
                        .replace('None','null')
                        .replace('True','true')
                        .replace('False','false'))
        # cleaned_text = re.sub(r'(\w+)"(\w+)', r"\1'\2", cleaned_text)
        
        return cleaned_text

    def deserialize(self, text):    
        deserialized_texts = []
        extract_dicts = re.findall(r'{.*?}',text)
        
        for str_dict in extract_dicts:
            cleaned_text = self.clean_text(str_dict)

            pattern = r'datetime.datetime\(.*\)'
            cleaned_text = re.sub(f'({pattern})',r'"\1"',cleaned_text)
            
            deserialized_text = json.loads(cleaned_text)
            
            if deserialized_text['created']!=None:
                deserialized_text['created'] = eval(deserialized_text['created'])
            
            deserialized_texts.append(deserialized_text)

        return deserialized_texts
        
    def extract_quoted_tweet(self, tweet):
        if type(tweet)!=float:
            text = re.findall(r"'rawContent': '?(.*?)'?, 'renderedContent'",tweet)[0]
            name = re.findall(r"'user': {'username': '?(.*?)'?,",tweet)[0]
            result = pd.Series({'quoted_text':text, 'quoted_username':name})
        else:
            result = pd.Series({'quoted_text':None, 'quoted_username':None})
            
        return result

    def create_new_features(self, mod_df):
        encoder = OrdinalEncoder()
        mod_df['sourceLabel_encoded'] = encoder.fit_transform(mod_df['sourceLabel'].values.reshape(-1, 1))
        
        encoder = OneHotEncoder()
        mod_df[encoder.categories_[0]] =  encoder.fit_transform(mod_df[['sourceLabel']]).toarray()
        
        binary_transform = (lambda column: [0 if type(tweet)==float else 1 for tweet in column])
        mod_df['isReplied']   = binary_transform(mod_df['inReplyToUser'])
        mod_df['isMentioned'] = binary_transform(mod_df['mentionedUsers'])
        
        mod_df['mentions'] = mod_df['rawContent'].apply(lambda x : re.findall(r'(@[^\s]+)', x))
        mod_df['mentionsCount'] = mod_df['rawContent'].str.count(r'@[\w\d]+')

        mod_df['charCount'] = mod_df['rawContent'].apply(lambda x: len(x))
        mod_df['mentionedUsers'] = mod_df['mentionedUsers'].apply(lambda x: self.deserialize(x) if type(x)==str else None)
        
        return mod_df

    def transform(self):
        mod_df = self.tweets_df.copy()
        mod_df = (mod_df[mod_df['lang']=='en']
                        .drop(['id','url','source','sourceUrl'], axis=1)                 
                        .reset_index(drop=True)
                        .copy())

        mod_df = mod_df.drop(['lang'], axis=1)


        #mod_df = mod_df.drop(['sourceLabel','inReplyToUser','mentionedUsers'], axis=1)
        extracted_df = pd.DataFrame([*mod_df['user'].apply(lambda x: self.extract_dict(x, True))])
        mod_df = (pd.concat([mod_df, extracted_df], axis=1)
                  .drop(['user','username','id','displayname','verified','created',
                         'location','protected','profileImageUrl','profileBannerUrl',
                         'rawDescription','renderedDescription','favouritesCount',
                         'friendsCount','mediaCount','statusesCount','inReplyToTweetId'], axis=1))


        # Converting columns containing numbers to int after extraction.
        for column in mod_df:
            if 'Count' in column:
                mod_df[column] = mod_df[column].astype('Int64').copy()
            
            
        mod_df = self.create_new_features(mod_df)
        mod_df = mod_df.drop(['links','media','link','inReplyToUser', 'descriptionLinks',
                              'renderedContent','conversationId','sourceLabel','mentions',
                              'cashtags','vibe'], axis=1)
        
        object_features = mod_df[mod_df.dtypes[mod_df.dtypes==object].index].copy()
        object_features['mentionsCount'] = object_features['mentionedUsers'].apply(lambda x: len(x) if x!=None else 0)
        object_features = object_features.drop(['mentionedUsers','label'], axis=1)
        extracted_quoted_tweets = object_features['quotedTweet'].apply(lambda x: self.extract_quoted_tweet(x))
        object_features = pd.concat([object_features, extracted_quoted_tweets], axis=1)

        # mergins cleaned object cols
        mod_df[object_features.columns] = object_features.copy()
        mod_df = mod_df.drop(['mentionedUsers','label','quotedTweet'], axis=1)
        
        return mod_df
