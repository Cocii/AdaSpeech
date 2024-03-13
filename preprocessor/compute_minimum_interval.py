import os
from tqdm import tqdm
with open("/workspace/nartts/AdaSpeech/preprocessed_data/cctv_212018_yw/wav_segment_time.txt", 'r') as f:
    lines = f.readlines()
with open("/workspace/nartts/AdaSpeech/preprocessed_data/cctv_212018_yw/valid_list.txt", 'r') as f:
    valid_lines = f.readlines()

valid_lines = [line.strip() for line in valid_lines]
valid_lines = [line for line in tqdm(lines) if line.split('|')[0] in valid_lines]
mini = 100
for line in tqdm(valid_lines):
    start = line.split('|')[1]
    end = line.split('|')[2]
    diff = float(end) - float(start)
    if diff < mini:
        mini = diff
print(mini)