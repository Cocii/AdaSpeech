import os
import yaml
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import argparse
import sys
sys.path.append('/workspace/nartts/AdaSpeech')
from text import _clean_text

def prepare_align(config):
    in_dir = config["path"]["raw_path"]
    out_dir = config["path"]["main_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    

    bad_sample_list_path = "/data/speech_data/LibriTTS_R/train-other-500_bad_sample_list.txt"
    # cd /data/speech_data/LibriTTS_R
    with open(bad_sample_list_path, "r") as f:
        i = 0
        for line in f.readlines():
            audio_name = line.strip("\n")
            if os.path.exists(audio_name):
                i += 1
                os.remove(audio_name)
        print("remove {} bad samples".format(i))

    i = 0
    # put all audios in one dir
    for speaker in tqdm(os.listdir(in_dir)):
        for chapter in os.listdir(os.path.join(in_dir, speaker)):
            for file_name in os.listdir(os.path.join(in_dir, speaker, chapter)):
                i += 1
                if file_name[-4:] != ".wav":
                    continue
                base_name = file_name[:-4]
                text_path = os.path.join(
                    in_dir, speaker, chapter, "{}.normalized.txt".format(base_name)
                )
                wav_path = os.path.join(
                    in_dir, speaker, chapter, "{}.wav".format(base_name)
                )
                if not (text_path and wav_path):
                    continue
                
                with open(text_path) as f:
                    text = f.readline().strip("\n")
                text = _clean_text(text, cleaners)
                wav, _ = librosa.load(wav_path, sr = sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value

                os.makedirs(os.path.join(out_dir), exist_ok=True)

                wavfile.write(
                    os.path.join(out_dir, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                    os.path.join(out_dir, "{}.lab".format(base_name)),
                    "w",
                ) as f1:
                    f1.write(text)
    print("total {} samples".format(i))

def main(config):
    prepare_align(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type = str, help = "path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader = yaml.FullLoader)
    main(config)