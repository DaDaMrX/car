'''
Send request to server for test
Author: Heng-Da Xu <dadamrxx@gmail.com>
Date  : 3/4/2019
'''
import requests
from pprint import pprint
import json


url = 'http://127.0.0.1:443/api/nlu'
data = {
    'model': 'rnn',
    'text': '打开空调',
}
r = requests.post(url, json=data)
pprint(json.loads(r.text))
