import os
import json
import torch
import numpy as np
from model import AdaSpeech, ScheduledOptim, AdaSpeech_spkr_en, AdaSpeech_spkr_en_cn, AdaSpeech_xlsr_en_cn
from model.discriminator import Discriminator

def get_disc(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    d_model = Discriminator(
        time_lengths=[32, 64, 128][:model_config['disc']['disc_win_num']],
        freq_length=preprocess_config['preprocessing']['mel']['n_mel_channels'], 
        hidden_size=model_config['disc']['mel_disc_hidden_size'], 
        kernel=(3, 3),
        norm_type=model_config['disc']['disc_norm'], 
        reduction=model_config['disc']['disc_reduction']
    ).to(device)

    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],'disc',
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        d_model.load_state_dict(ckpt['model'])
    if train:
        scheduled_optim = ScheduledOptim(
            d_model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        d_model.train()
        return d_model, scheduled_optim
    
    d_model.eval()
    d_model.requires_grad_ = False
    return d_model

def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    separated = model_config["language_speaker"]["separated"]
    spk_type = train_config["spk-type"]
    if separated:
        model = AdaSpeech(preprocess_config, model_config).to(device)
    else:
        if spk_type == "spk":
            if preprocess_config["preprocessing"]["text"]["language"] == ["en", "cn"]:
                model = AdaSpeech_spkr_en_cn(preprocess_config, model_config).to(device)
            if preprocess_config["preprocessing"]["text"]["language"] == ["en"]:
                model = AdaSpeech_spkr_en(preprocess_config, model_config).to(device)
                src_word_emb_weight = model.encoder.src_word_emb.weight
                temp_src_word_emb_weight = torch.empty(361, 256)
                temp_src_word_emb_weight[:src_word_emb_weight.size(0), :] = src_word_emb_weight[:361, :]
                new_src_word_emb_weight = torch.nn.Parameter(temp_src_word_emb_weight)           
                model.encoder.src_word_emb.weight = new_src_word_emb_weight
            else:
                pass
        if spk_type == "xlsr" and preprocess_config["preprocessing"]["text"]["language"] == ["en", "cn"]:
            model = AdaSpeech_xlsr_en_cn(preprocess_config, model_config).to(device)
        else:
            pass
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)

        # print("Current model state dict keys and sizes:")
        # for name, param in model.state_dict().items():
        #     print(name, param.size())

        # print("Loaded model state dict keys and sizes:")
        # for name, param in ckpt["model"].items():
        #     print(name, param.size())

        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model

def load_pretrain(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    separated = model_config["language_speaker"]["separated"]
    spk_type = model_config["spk-type"]
    if separated:
        model = AdaSpeech(preprocess_config, model_config).to(device)
    else:
        if spk_type == "spk":
            if preprocess_config["preprocessing"]["text"]["language"] == ["en", "cn"]:
                model = AdaSpeech_spkr_en_cn(preprocess_config, model_config).to(device)
            if preprocess_config["preprocessing"]["text"]["language"] == ["en"]:
                model = AdaSpeech_spkr_en(preprocess_config, model_config).to(device)
                src_word_emb_weight = model.encoder.src_word_emb.weight
                temp_src_word_emb_weight = torch.empty(361, 256)
                temp_src_word_emb_weight[:src_word_emb_weight.size(0), :] = src_word_emb_weight[:361, :]
                new_src_word_emb_weight = torch.nn.Parameter(temp_src_word_emb_weight)           
                model.encoder.src_word_emb.weight = new_src_word_emb_weight
            else:
                pass
        if spk_type == "xlsr" and preprocess_config["preprocessing"]["text"]["language"] == ["en", "cn"]:
            model = AdaSpeech_xlsr_en_cn(preprocess_config, model_config).to(device)
        else:
            pass

    ckpt_path = args.pretrain_dir
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model"])
    scheduled_optim = ScheduledOptim(
        model, train_config, model_config, 0
    )
    
    for param in model.named_parameters():
        if "layer_norm" not in param[0]:
            param[1].requires_grad = False
        if "encoder" in param[0]:
            param[1].requires_grad = False
        if "variance_adaptor" in param[0]:
            param[1].requires_grad = False
        if "UtteranceEncoder" in param[0]:
            param[1].requires_grad = False
        if "PhonemeLevelEncoder" in param[0]:
            param[1].requires_grad = False
        if "PhonemeLevelPredictor" in param[0]:
            param[1].requires_grad = False
        if "speaker_emb" in param[0]:
            param[1].requires_grad = True
    model.train()
    return model, scheduled_optim


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)
        elif name == "BigVGAN":
            wavs = vocoder(mels).squeeze(1)
    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs