import json
import os
import numpy as np
from tqdm import tqdm
import argparse
import yaml
from sklearn.preprocessing import StandardScaler
import re

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon
lexicon = read_lexicon(os.path.join("/workspace/nartts/AdaSpeech/lexicon", "pinyin-lexicon-r.dict"))
INITIALS = [
    'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'zh', 'ch', 'sh',
    'r', 'z', 'c', 's', 'j', 'q', 'x', 'y', 'w'
]

FINALS = [
    'io',
    'ueng'
]
TONES = ["1","2","3","4","5"]

for i in INITIALS:
    for f in FINALS:
        for t in TONES:
            combination = i + f + t + " " + i + " "+ f + t +"\n"
            with open("/workspace/nartts/AdaSpeech/lexicon/mandarin_add.dict", "a") as file:
                file.write(combination)
