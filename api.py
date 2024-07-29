import requests
import json
from dotenv import load_dotenv
import os
load_dotenv()
# 7DWY5L6LOZDTD0QA
#7DWY5L6LOZDTD0QA
def collect_data(currency):
    url = f'https://min-api.cryptocompare.com/data/v2/histohour'
    params = {
        'fsym': currency,
        'tsym': 'USD',
        'limit': 720,
        'aggregate': 1
    }
    headers = {
        'authorization':os.getenv('API')
    }

    response=requests.get(url,params=params)
    result=response.json()
    return result
