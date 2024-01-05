from paddlenlp import Taskflow
from pprint import pprint

import requests
from bs4 import BeautifulSoup

import matplotlib.pyplot as plt

# 1. request

# course_id = str(input())
course_id = '159'

url = "https://nces.cra.moe/course/" + course_id
response = requests.get(url)
content = response.text

soup = BeautifulSoup(content, 'lxml')

paragraphs = [p.text for p in soup.find_all('p')]

with open('extract.txt', 'w') as f:
    for paragraph in paragraphs:
        f.write(paragraph + '\n')

# 2. nlp

schema = {'评价维度': ['观点词', '情感倾向[正向，负向]']}
ie = Taskflow('information_extraction', schema=schema)

with open('extract.txt', 'r') as f:
    content = f.read()

result = ie(content)[0]

pprint(result)

# 3. analysis

items = result['评价维度']

for (index, item) in enumerate(items):
    name = item['text']
    attitude = item['relations']['情感倾向[正向，负向]'][0]['text']
    attitude_pos = item['relations']['情感倾向[正向，负向]'][0]['probability']
    predictions = [(words['text'], words['probability']) for words in item['relations']['观点词']]
    print('第 %d 条结果: %s 总体评价：%f 概率的 %s' % (index, name, attitude_pos, attitude))
    for prediction in predictions:
        print('观点词: %s, 概率: %f' % prediction)
    print()

