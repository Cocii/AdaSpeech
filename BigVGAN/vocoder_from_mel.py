import argparse
import torch
import json
import numpy as np
import sys
sys.path.append("/workspace/nartts/AdaSpeech")
from utils.tools import to_device, synth_samples, AttrDict
import os
from librosa.util import normalize
# sys.path.append("vocoder")
# from vocoder.models.hifigan import Generator
from vocoder.models.BigVGAN import BigVGAN as Generator
from scipy.io import wavfile
import yaml
from utils.tools import pad_1D, pad_2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_vocoder(config, checkpoint_path):
    config = json.load(open(config, 'r', encoding='utf-8'))
    config = AttrDict(config)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    vocoder = Generator(config).to(device).eval()
    vocoder.load_state_dict(checkpoint_dict['generator'])
    vocoder.remove_weight_norm()
    return vocoder

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mel_path", type=str, default=None, required= True, help="path to mel path"
    )
    parser.add_argument(
        "--vocoder_checkpoint", type=str, default=None, required= True, help="path to vocoder checkpoint"
    )
    parser.add_argument(
        "--vocoder_config", type=str, default=None, required=True, help="path to vocoder config"
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    args = parser.parse_args()
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    mel_path = args.mel_path
    # mel_path = os.path.join(
    #         p_path,
    #         "mel",
    #         "{}-mel-{}.npy".format(speaker, basename),
    #     )
    mel = np.load(mel_path)
    # print("mel.shape: ", mel.shape)
    
    form_mel_preprocessor = 1 # mel spectrum for preprocessing session
    
    mel = torch.from_numpy(mel).float().to(device)
    mel = mel.T

    print("mel.size(): ", mel.size())
    if form_mel_preprocessor:
        mel = mel.detach().transpose(0, 1)
        mel = mel.unsqueeze(0)


    from utils.model import vocoder_infer
    vocoder = get_vocoder(args.vocoder_config, args.vocoder_checkpoint)
    wav = vocoder_infer(mel, vocoder, model_config, preprocess_config)[0]
    # print("wav.shape(): ", wav.shape())
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    # output path
    out_path = "/data/speech_data/ref_audios/"
    # out_path = os.path.dirname(args.mel_path.split)
    # print("out_path: ",out_path)
    file_name = args.mel_path.split(".")[0].split("/")[-1]
    # print("file_name: ", file_name)
    basename = out_path + file_name + ".wav"
    print("Saving \"{}\"...".format(basename))
    wavfile.write(os.path.join(basename), sampling_rate, wav)

    