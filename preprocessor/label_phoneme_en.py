import os
import re

import os
from pypinyin import pinyin, Style
import sys
sys.path.append("/workspace/nartts/AdaSpeech")
from text.cleaners import english_cleaners
from text import text_to_sequence
import numpy as np
import yaml
from tqdm import tqdm
sys.path.append("/root/Documents/MFA/pretrained_models/g2p/english_us_arpa.zip")
# from g2p_en import english_us_arpa as G2p
from g2p_en import G2p

def is_chinese_char(char):
    # including punctuation
    patten = r"[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]"
    if char is None or len(char) == 0 or len(char) > 1:
        return False
    if 0x4e00 <= ord(char) <= 0x9fff or re.search(patten,char):
        return True
    else:
        return False
    
def read_pinyin(lex_path):
    lexicon = {}
    lines = []
    with open(lex_path) as f:
        for line in f:
            lines.append(line)
        for line in reversed(lines):
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def read_lexicon(lex_path):
    lexicon = {}
    lines = []
    with open(lex_path) as f:
        for line in f:
            lines.append(line)
        for line in reversed(lines):
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = []
            lexicon[word.lower()].append(phones)
            # print("Loaded  \"{}\" lexicon {}.".format(word.lower(), lexicon[word.lower()]))
    # print(type(lexicon))
    # print("it in lexicon: ", lexicon['is'])
    return lexicon

def preprocess_en_cn(phones_origin, text, lexicon, g2p):
    text = re.sub(r'([\^\*&%#@.,\-?!;:"\'()<>[\]{}…=+~\\$|/]|[\s]|[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5])+$', '', text)
    words = re.split(r'([\^\*&%#@,?!;:"()<>[\]{}…=+~\\$|/]|[\s]|[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]|[\u4e00-\u9fa5])', text)
    # 中文 [\u4e00-\u9fa5]
    # 中文标点 [\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]
    print("text: ", text)
    # print("\nwords : ", words)
    phones_origin = phones_origin.split()
    print("phones_origin: ", phones_origin)
    abs_len = len(phones_origin)
    count_len = 0
    phones = []
    labels = []
    for w in words:
        if w is not None:
            tmp_cleaned = re.split(r'[\-_.\s]',english_cleaners(w))
            tem_cleaned = [tmp for tmp in tmp_cleaned if tmp != "" and tmp not in {',', '?', '!', '=','+','-','_','\\','|','.','{','}','\'','/','<','>',';',':','[',']','`','~'}]
            # print("tmp_cleaned: ", tem_cleaned)
            for tmp in tem_cleaned:
                if tmp.lower() in lexicon:
                    phones_lexicon = lexicon[tmp.lower()]
                    ifmatch = False
                    for p in phones_lexicon:
                        phones_tem = p
                        phones_len = len(phones_tem)
                        if phones_origin[count_len] == 'sp':
                            labels += ['S']
                            phones += ['sp']
                            count_len += 1
                        # print("p: ", p, ", len(p): ", len(p))
                        # print("two phones: ", phones_tem, phones_origin[count_len:count_len+phones_len])
                        # if len(phones_tem) == len(phones_origin[count_len:count_len+phones_len]):
                        if phones_tem == phones_origin[count_len:count_len+phones_len]:
                            print("0 two phones: ", phones_tem, phones_origin[count_len:count_len+phones_len])
                            # print("yes")
                            phone = phones_origin[count_len:count_len+phones_len]
                            count_len += phones_len
                            ifmatch = True
                            break
                        if count_len > 0 and phones_tem == phones_origin[count_len-1:count_len+phones_len-1] and phones_origin[count_len-1] in {'T','P','D'}:
                            print("1 two phones: ", phones_tem, phones_origin[count_len-1:count_len+phones_len-1])
                            phone = phones_origin[count_len:count_len+phones_len-1]
                            count_len -= 1
                            count_len += phones_len
                            ifmatch = True
                            break

                    assert ifmatch == True

                    if len(phone) == 0:
                        print("len(phone): 0", phone)
                    # assert len(phone) != 0 
                    elif len(phone) == 1:
                        label = ["S"]
                    else:
                        label = ["M" for _ in phone]
                        label[0] = "B"
                        label[-1] = "E"    
                    phones += phone
                    labels += label
                else:
                    phone = list(filter(lambda p: p != " ", g2p(tmp.lower())))
                    # print("not in lexicon: ", phone)
                    if len(phone) == 0:
                        phone = []
                        label = []
                    elif len(phone) == 1:
                        if phone == [phones_origin[count_len]]:
                            print("tmp.lower(), phone: ", tmp.lower(), phone)
                            label = ["S"]
                    else:
                        if phone[0] == phones_origin[count_len] and phone[-1] == phones_origin[count_len:count_len + len(phone)][-1]:
                            print("2 two phones: ", phone, phones_origin[count_len:count_len + len(phone)])
                            label = ["M" for _ in phone]
                            label[0] = "B"
                            label[-1] = "E"    
                    phones += phone
                    labels += label
    print("abs_len: ", abs_len)
    print("labels: ", len(labels), labels)
    print("phones: ", len(phones), phones)
    assert abs_len == len(labels)
    # print("\nwords : ", phones)

    phones = "}{".join(phones)
    # print("phones before: ", phones)
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    # print("phones before: ", phones)
    # for p in phones:

    phones = phones.replace("}{", " ")
    # print("Raw Text Sequence: {}".format(text))
    # print("Phoneme Sequence: {}".format(phones))
    return phones, labels



work_path = "/workspace/nartts/AdaSpeech/preprocessed_data/preprocessed_data_libri_spkr/"
train_path = os.path.join(work_path,"train_origin.txt")
val_path = os.path.join(work_path,"val_origin.txt")
val_path_write = os.path.join(work_path,"val_phoneme_label.txt")
train_path_write = os.path.join(work_path,"train_phoneme_label.txt")

lines = []
# if it's combined
for i, path in enumerate([val_path]):
    with open(path,"r",encoding="utf-8") as f:
        lines.append(f.readlines())

preprocess_config = yaml.load(
        open( "/workspace/nartts/AdaSpeech/config/en_cn_spkr/preprocess.yaml", "r"), Loader = yaml.FullLoader
    )
# lexicon = read_lexicon(os.path.join(preprocess_config["path"]["lexicon_path"], "english_us.dict"))
lexicon = read_lexicon(os.path.join("/data/speech_data/LibriTTS/english_us.dict"))
# lexicon_pinyin = read_pinyin(os.path.join(preprocess_config["path"]["lexicon_path"], "pinyin-lexicon-r.dict"))

g2p = G2p()
count = 0
total = 0
for i, path in enumerate([val_path_write]):
    newline = []
    with open(path, "w", encoding = "utf-8") as f:
        for line in tqdm(lines[i]):
            line = line.strip("\n")
            line = line.split("|")
            l2s = line[2].strip("}").strip("{") # phonemes
            # print("l2s: ", l2s)
            l3s = line[3] # rawtext
            phonemes, labels = preprocess_en_cn(l2s, l3s, lexicon, g2p)
            # control the path
            newline = (line[0]+"|"+line[1]+"|"+ "{" + "".join(phonemes) + "}" + "|" + "".join(l3s) + "|" + "0" + "|" + " ".join(labels))
            l2s = l2s.split(" ")
            l2s = [l for l in l2s if l != "sp"]
            l2s = ' '.join(l2s)

            phonemes = phonemes.split(" ")
            phonemes = [l for l in phonemes if l != "sp"]
            phonemes = ' '.join(phonemes)
            # print("if the phonemes == l3s", len(phonemes) == len(l2s))
            if len(phonemes) == len(l2s):
                count += 1
            print("phonemes :", phonemes)
            print("line[2] : ", l2s)
            # print(newline)
            f.write(newline+"\n")
            total += 1
print(count, " same phones!", "/ total ", total, "samples")
        