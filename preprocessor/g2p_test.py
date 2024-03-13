import os
from pypinyin import pinyin, Style
import sys
import numpy as np
import yaml
from tqdm import tqdm
from g2p_en import G2p
g2p = G2p()

java_list0 = [
    'mybatis',
    'hibernate',
    'TestNG',
    'xml',
    'Junit',
    'Mockito',
    'mock',
    'maven',
    'surfire',
    'jacoco',
    'memcache',
    'redis',
    'rabbitmq',
    'kafka',
    'tomcat',
    'jetty',
    'servlet',
    'zookeeper',
    'Logback',
    'Paxos',
    'Servless',
    'Nginx',
    'MYSQL',
    'HTTP',
    'SQL',
    'CSRF',
    'XSS',
    'LOG4J',
    'SLF4J',
    'API',
    'DDL',
    'DML',
    'TCC',
    'SOA',
    'MVC',
    'RBAC',
    'OAuth',
    'eBay',
]
java_list1 = [
    'mybatis',
    'hibernate',
    'Test N G',
    'x m l',
    'Junit',
    'Mockito',
    'mock',
    'maven',
    'sur fire',
    'jacoco',
    'mem cache',
    'redis',
    'rabbit m q',
    'kafka',
    'tomcat',
    'jetty',
    'servlet',
    'zookeeper',
    'Log back',
    'Paxos',
    'Servless',
    'Nginx',
    'MY S Q L',
    'H T T P',
    'S Q L',
    'C S R F',
    'X S S',
    'LOG 4 J',
    'S L F 4 J',
    'A P I',
    'D D L',
    'D M L',
    'T C C',
    'S O A',
    'M V C',
    'R B A C',
    'O Auth',
    'e Bay',
]

python_list0 = [
    'pytorch',
    'tensorflow',
    'Scrapy',
    'CPython',
    'mutator',
    'coroutine'

]
python_list1 = [
    'pytorch',
    'tensor flow',
    'Scrapy',
    'C Python',
    'mutator',
    'co routine'

]
print('java_list: ', ', ,'.join(java_list1))
print('python_list: ', ', ,'.join(python_list1))
# for g in tqdm(python_list1):
#     print("Origin: ['{}']".format(g), "Arpabet: ", g2p(g))

for i,j in enumerate(java_list1):
    print("(\'{}\', \'{}\'),".format(java_list0[i], java_list1[i]))

for i,j in enumerate(python_list1):
    print("(\'{}\', \'{}\'),".format(python_list0[i], python_list1[i]))

