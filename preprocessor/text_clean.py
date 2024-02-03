import os
import re
import sys
sys.path.append('/workspace/nartts/AdaSpeech')
from text import _clean_text
from tqdm import tqdm
dir_path = "/data/speech_data/LibriTTS/audios"

file_names = os.listdir(dir_path)
file_names = [file_name for file_name in file_names if file_name.endswith(".lab")]
cleaners = ["english_cleaners"]

for file_name in tqdm(file_names):
    if file_name.endswith(".lab"):
        with open(os.path.join(dir_path, file_name), "r") as f:
            text = f.readline().strip("\n")
        text = _clean_text(text, cleaners)
        with open(os.path.join(dir_path, file_name), "w") as f:
            f.write(text)