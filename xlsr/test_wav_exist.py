import os
from tqdm import tqdm
import librosa
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


if __name__ == "__main__":
    input_path = "/workspace/nartts/AdaSpeech/preprocessed_data/libri_spkr/train.txt"
    # input_path = "/workspace/nartts/AdaSpeech/preprocessed_data/cctv_212018_yw/train.txt"
    with open(input_path, 'r') as f:
        lines = f.readlines()
    ids = []
    speakers = []
    phones = []
    raws = []
    lgs = []
    for l in tqdm(lines):
        id, speaker, phone, raw, lg = l.split('|')
        ids.append(id)
        speakers.append(speaker)
        phones.append(phone)
        raws.append(raw)
        lgs.append(lg)
    
    if input_path.split('/')[-2] == "libri_spkr":
        # all_wavs = libri_get_wavs("/data/speech_data/LibriTTS/audios/en/")
        all_xlsr = libri_get_xlsr("/workspace/nartts/AdaSpeech/preprocessed_data/libri_spkr/xlsr/")
    if input_path.split('/')[-2] == "cctv_212018_yw":
        all_wavs, all_wavs_no_root = cctv_get_wavs("/data/speech_data/cctv_cjy/")
        all_xlsr = cctv_get_xlsr("/workspace/nartts/AdaSpeech/preprocessed_data/cctv_212018_yw/xlsr/")

    with open("/data/speech_data/cuijiayan/tools/xlsr/text.txt", "a") as f:
        # for wav in tqdm(all_wavs):
        #     if os.path.getsize(wav) > 2560000:
        #         f.write(wav + '\n')
        for i in tqdm(range(len(ids))):
            if (speakers[i] + '-xlsr-'+ ids[i] + '.npy') in all_xlsr: # no root
                pass
            else:
                if input_path.split('/')[-2] == "libri_spkr":
                    f.write("/data/speech_data/LibriTTS/audios/en/" + speakers[i] + '_' + ids[i] + '.wav' + '\n')
                if input_path.split('/')[-2] == "cctv_212018_yw":
                    f.write(all_wavs[all_wavs_no_root.index(ids[i]+'.wav')] + '\n')

