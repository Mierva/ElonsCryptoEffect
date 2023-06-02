from concurrent.futures import ThreadPoolExecutor, as_completed
import snscrape.modules.twitter as sntwitter
import pandas as pd

class TweetScraper:    
    def __init__(self, start='2023-04-11', end='2023-04-13', freq='d', maxEmptyPages=20, max_workers=8):         
        self.dateRange = pd.date_range(start=start, end=end, freq=freq)
        self.maxEmptyPages = maxEmptyPages 
        self.maxWorkers = max_workers

        self.queries = [f"from:elonmusk since:{d1.strftime('%Y-%m-%d')} until:{d2.strftime('%Y-%m-%d')}"
                        for d1, d2 in zip(self.dateRange, self.dateRange[1:])]

    def sequent_download_tweets(self, query):
        tweets = []
        for tweet in sntwitter.TwitterSearchScraper(query=query, maxEmptyPages=self.maxEmptyPages).get_items():
            tweets.append(tweet)
        return tweets

    def parallel_download_tweets(self):
        tweets_list = []
        with ThreadPoolExecutor(max_workers=self.maxWorkers) as executor:
            futures = [executor.submit(self.sequent_download_tweets, query) for query in self.queries]
                
            # Append results to the tweets list
            for future in as_completed(futures):
                tweets_list += future.result()
                
        return tweets_list