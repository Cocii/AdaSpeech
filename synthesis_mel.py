import argparse
from email.generator import Generator
import json
import os
from sklearn.utils import shuffle

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_param_num
from utils.tools import to_device, log, synth_one_sample, AttrDict
from model import AdaSpeechLoss
from dataset import Dataset

from evaluate import evaluate
import sys
sys.path.append("vocoder")
# from vocoder.models.hifigan import Generator
from vocoder.models.BigVGAN import BigVGAN as Generator
import numpy as np
import soundfile as sf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def synth_one_sample_save(targets, predictions, vocoder, model_config, preprocess_config):
    audios_path = os.path.join("/data/speech_data/libri_cctv_vocoder_fintune/", "wavs/")
    mels_path = os.path.join("/data/speech_data/libri_cctv_vocoder_fintune/", "mels/")
    for i in range(len(targets[0])):
        basename = targets[0][i]
        speakernames = targets[2][i]
        mel_len = predictions[10][i].item()
        mel_target = targets[6][i, :mel_len].detach().transpose(0, 1)
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        tmpname = speakernames+"_"+basename
        if vocoder is not None:
            from utils.model import vocoder_infer

            wav_reconstruction = vocoder_infer(
                mel_target.unsqueeze(0),
                vocoder,
                model_config,
                preprocess_config,
            )[0]
            wav_prediction = None
        else:
            wav_reconstruction = wav_prediction = None
        sampling_rate = preprocess_config["preprocessing"]["audio"][
            "sampling_rate"
        ]
        audio_normalized = wav_reconstruction / np.max(np.abs(wav_reconstruction))
        
        
        audio_path = os.path.join(audios_path, tmpname+".wav")
        mel_path = os.path.join(mels_path, tmpname+".npy")

        if not os.path.exists(audio_path):
            sf.write(audio_path, audio_normalized, sampling_rate)
        else:
            # print(audio_path," already exist!")
            file = open("/workspace/nartts/AdaSpeech/error_files.txt", "a") 
            file.write(mel_path + "\n")

        if not os.path.exists(mel_path):
            np.save(
                    os.path.join(mel_path),
                    mel_prediction.cpu().numpy(),
                )
        else:
            file = open("/workspace/nartts/AdaSpeech/error_files.txt", "a") 
            file.write(mel_path + "\n")
            # print(mel_path," already exist!")

    return wav_reconstruction, wav_prediction, speakernames+"_"+basename

def get_vocoder(config, checkpoint_path):
    config = json.load(open(config, 'r', encoding='utf-8'))
    config = AttrDict(config)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    vocoder = Generator(config).to(device).eval()
    vocoder.load_state_dict(checkpoint_dict['generator'])
    vocoder.remove_weight_norm()
    return vocoder

def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset

    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        shuffle = True,
        batch_size=batch_size * group_size,
        collate_fn=dataset.collate_fn,
        num_workers=4,
    )

    # Prepare model
    #
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = AdaSpeechLoss(preprocess_config, model_config).to(device)
    print("Number of AdaSpeech Parameters:", num_param)

    # Load vocoder
    #vocoder = get_vocoder(model_config, device)
    vocoder = get_vocoder(args.vocoder_config, args.vocoder_checkpoint)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    total_step = train_config["step"]["total_step"]
    synth_step = 1
    phoneme_level_encoder_step = train_config["step"]["phoneme_level_encoder_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()
    
    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                # Forward
                if step >= phoneme_level_encoder_step:
                    phoneme_level_predictor = True
                    exe_batch = batch + (phoneme_level_predictor, )
                    output = model(*(exe_batch))
                else:
                    phoneme_level_predictor = False
                    exe_batch = batch + (phoneme_level_predictor, )
                    output = model(*(exe_batch))
                    
                if step % synth_step == 0:
                    wav_reconstruction, wav_prediction, tag = synth_one_sample_save(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--vocoder_checkpoint", type=str, default=None, required= True, help="path to vocoder checkpoint"
    )
    parser.add_argument(
        "--vocoder_config", type=str, default=None, required=True, help="path to vocoder config"
    )

    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)


    main(args, configs)
