from concurrent.futures import ThreadPoolExecutor, as_completed
import snscrape.modules.twitter as sntwitter
import pandas as pd

class TweetScraper:    
    def __init__(self, start='2023-04-11', end='2023-04-13', freq='d'):         
        self.date_range = pd.date_range(start=start, end=end, freq=freq)

        self.queries = [f"from:elonmusk since:{d1.strftime('%Y-%m-%d')} until:{d2.strftime('%Y-%m-%d')}"
                        for d1, d2 in zip(self.date_range, self.date_range[1:])]

    def sequent_download_tweets(self, query):
        tweets = []
        for tweet in sntwitter.TwitterSearchScraper(query).get_items():
            tweets.append(tweet)
        return tweets

    def parallel_download_tweets(self):
        tweets_list = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.sequent_download_tweets, query) for query in self.queries]
                
            # Append results to the tweets list
            for future in as_completed(futures):
                tweets_list += future.result()
                
        return tweets_list