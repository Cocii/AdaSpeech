import os
from tqdm import tqdm
import librosa
import numpy as np
def modify_train(input_path, target_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    speakers = []
    ids = []
    for line in tqdm(lines):
        speakers.append(line.split('|')[0])
        ids.append(line.split('|')[1])
    short_mel_list = [speakers[i]+ids[i] for i in range(len(speakers))]

    print(input_path.split('/')[-1], ", short_mel_list length: ", len(short_mel_list))

    with open(os.path.join(target_path, "train.txt"), 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    print("lines length before: ", len(lines))

    result = []
    for i in tqdm(range(len(lines))):
        if lines[i].split('|')[1]+lines[i].split('|')[0] in short_mel_list:
            pass
        else:
            result.append(lines[i] + '\n')
    print("lines length after: ", len(result))

    target_train_path = os.path.join(target_path, "train_remove_short.txt")
    # os.makedirs(target_train_path, exist_ok=True)
    with open(target_train_path, 'w') as f:
        f.writelines(result)
    
def modify_val(input_path, target_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    speakers = []
    ids = []
    for line in tqdm(lines):
        speakers.append(line.split('|')[0])
        ids.append(line.split('|')[1])
    short_mel_list = [speakers[i]+ids[i] for i in range(len(speakers))]

    print(input_path.split('/')[-1], ", short_mel_list length: ", len(short_mel_list))

    with open(os.path.join(target_path, "val.txt"), 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    print("lines length before: ", len(lines))

    result = []
    for i in tqdm(range(len(lines))):
        if lines[i].split('|')[1]+lines[i].split('|')[0] in short_mel_list:
            pass
        else:
            result.append(lines[i] + '\n')
    print("lines length after: ", len(result))

    target_train_path = os.path.join(target_path, "val_remove_short.txt")
    # os.makedirs(target_train_path, exist_ok=True)
    with open(target_train_path, 'w') as f:
        f.writelines(result)



if __name__ == "__main__":
    input_path = "/workspace/nartts/AdaSpeech/preprocessor/short_mel_libri.txt"
    target_path = "/workspace/nartts/AdaSpeech/preprocessed_data/libri_spkr"

    # input_path = "/workspace/nartts/AdaSpeech/preprocessor/short_mel_cctv.txt"
    # target_path = "/workspace/nartts/AdaSpeech/preprocessed_data/cctv_212018_yw"

    modify_train(input_path, target_path)
    modify_val(input_path, target_path)
