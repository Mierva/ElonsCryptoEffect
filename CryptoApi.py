import requests

class CryptoApi:
    def __init__(self, token):
        self.mother_endpoint = "https://min-api.cryptocompare.com/data"      
        self.TOKEN = token
        self.HEADER = {'Authorization': self.TOKEN,
                       "Content-Type": 'application/json'}


    def fetch_data(self, crypto:str, currency:str, period:str, period_count:int, allData=0):
        '''Returns crypto summary for a given period in specified currency.

        ## Parameters:
            crypto: str
                BTC/ETH/DOGE etc.
            currency: str
                USD/EUR/UAH etc.
            period: str 
                day/hour/minute.
            period_count: int
                last n of a period (n days)
                n = 1 returns previous period and current
            allData (Optional): int
                1 - get all records.
                0 - get specified amount of period_count.
        ## Returns:
            dict: json containing request's response.
        '''        

        endpoint = f'{self.mother_endpoint}/v2/histo{period}?fsym={crypto}&tsym={currency}&limit={period_count}'
        response = requests.get(endpoint, params={'allData':allData}, headers=self.HEADER)
        
        return response.json()