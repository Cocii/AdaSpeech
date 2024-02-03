import re
from g2p_en import G2p
import os
from pypinyin import pinyin, Style
from text.cleaners import english_cleaners
from text import text_to_sequence
import numpy as np


def is_chinese_char(char):
    # including punctuation
    patten = r"[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]"
    if char is None or len(char) == 0 or len(char) > 1:
        return False
    if 0x4e00 <= ord(char) <= 0x9fff or re.search(patten,char):
        return True
    else:
        return False
    
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

def preprocess_en_cn(text, preprocess_config):
    text = re.sub(r'([\^\*&%#@.,\-?!;:"\'()<>[\]{}…=+~\\$|/]|[\s]|[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5])+$', '', text)
    lexicon = read_lexicon(os.path.join(preprocess_config["path"]["lexicon_path"], "english_us.dict"))
    lexicon_pinyin = read_lexicon(os.path.join(preprocess_config["path"]["lexicon_path"], "pinyin-lexicon-r.dict"))
    g2p = G2p()
    phones = []
    words = re.split(r'([\^\*&%#@,?!;:"()<>[\]{}…=+~\\$|/]|[\s]|[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]|[\u4e00-\u9fa5])', text)
    # 中文 [\u4e00-\u9fa5]
    # 中文标点 [\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]
    print("\nwords : ", words)
    for w in words:
        if w is not None:
            if is_chinese_char(w):
                p_pinyin = pinyin(w, style=Style.TONE3, strict=False, neutral_tone_with_five=True)
                p_pinyin = p_pinyin[0]
                p_pinyin = p_pinyin[0]
                print("w: ", w, " p_pinyin: ", p_pinyin)
                if p_pinyin in lexicon_pinyin:
                    phones += lexicon_pinyin[p_pinyin]
                else:
                    # if w in ['\uff0c', '\u3002']:
                    phones.append("sp")
            else:
                tmp_cleaned = re.split(r'[\-_.\s]',english_cleaners(w))
                tem_cleaned = [tmp for tmp in tmp_cleaned if tmp != ""]
                print("tmp_cleaned: ", tem_cleaned)
                for tmp in tmp_cleaned:
                    if tmp.lower() in lexicon:
                        phones += lexicon[tmp.lower()]
                    else:
                        phones += list(filter(lambda p: p != " ", g2p(tmp.lower())))

    print("\nwords : ", phones)

    phones = "{" + "}{".join(phones) + "}"
    # print("phones before: ", phones)
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    # print("phones before: ", phones)
    # for p in phones:

    phones = phones.replace("}{", " ")
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, []
        )
    )
    # print("sequence: ", sequence)
    return np.array(sequence)