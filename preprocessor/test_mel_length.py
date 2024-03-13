import os
from tqdm import tqdm
import librosa
import numpy as np
def cctv_get_wavs(root_dir):
    dirs = ['2018', '2020', '2021']
    subdirectories = []
    path_no_root = []
    for d in dirs:
        for root, mid, files in tqdm(os.walk(os.path.join(root_dir, d))):
            for file in files:
                if file.endswith(".wav"):
                    subdirectories.append(os.path.join(root, file))
                    path_no_root.append(os.path.join(file))
    return subdirectories, path_no_root

def libri_get_wavs(root_dir):
    subdirectories = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                subdirectories.append(os.path.join(root, file))
    return subdirectories


def libri_get_xlsr(root_dir):
    subdirectories = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".npy") and file.split('-')[1] == 'xlsr':
                subdirectories.append(os.path.join(file)) # no root
    return subdirectories

def cctv_get_xlsr(root_dir):
    subdirectories = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".npy") and file.split('-')[1] == 'xlsr':
                subdirectories.append(os.path.join(file)) # no root
    return subdirectories

def load_mel(root_dir):
    subdirectories = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".npy") and file.split('-')[1] == 'mel':
                subdirectories.append(os.path.join(file)) # no root
    return subdirectories

if __name__ == "__main__":
    # input_path = "/workspace/nartts/AdaSpeech/preprocessed_data/libri_spkr/mel"
    input_path = "/workspace/nartts/AdaSpeech/preprocessed_data/cctv_212018_yw/mel"
    
    mel_paths = load_mel(input_path)
    # numpy 128*80  os.path.getsize(mel) == 82048
    all_mels_length = [os.path.getsize(os.path.join(input_path, mel_path)) for mel_path in tqdm(mel_paths)]
    speakers = []
    ids = []
    for mel_path in mel_paths:
        speakers.append(mel_path.split('.')[0].split('-mel-')[0])
        ids.append(mel_path.split('.')[0].split('-mel-')[1])
    
    if input_path.split('/')[-2] == "libri_spkr":
        with open("/workspace/nartts/AdaSpeech/preprocessor/short_mel_libri.txt", "a") as f:
            for i in range(len(all_mels_length)):
                if all_mels_length[i] < 82048:
                    f.write(speakers[i] + '|' + ids[i] + '\n')
    if input_path.split('/')[-2] == "cctv_212018_yw":
        with open("/workspace/nartts/AdaSpeech/preprocessor/short_mel_cctv.txt", "a") as f:
            for i in range(len(all_mels_length)):
                if all_mels_length[i] < 82048:
                    f.write(speakers[i] + '|' + ids[i] + '\n')