import argparse

import torch
import yaml
import sys
import json
import librosa
import numpy as np
from dataset import TextDataset
from pypinyin import pinyin, Style
from utils.model import get_model
from utils.tools import to_device, synth_samples, AttrDict
from text.cleaners import english_cleaners
from torch.utils.data import DataLoader
from dataset import Dataset
from text import text_to_sequence
from datetime import datetime
from g2p_en import G2p
from string import punctuation
import audio as Audio
import os
import soundfile as sf
from librosa.util import normalize
from tqdm import tqdm

# sys.path.append("vocoder")
# from vocoder.models.hifigan import Generator
from vocoder.models.BigVGAN import BigVGAN as Generator
import re
from pyannote.audio import Inference
from pyannote.audio import Model
from frontend_en_cn import preprocess_en_cn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"], "english_us.dict")

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(os.path.join(preprocess_config["path"]["lexicon_path"], "pinyin-lexicon-r.dict"))
    
    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, []
        )
    )

    return np.array(sequence)



def get_vocoder(config, checkpoint_path):
    config = json.load(open(config, 'r', encoding='utf-8'))
    config = AttrDict(config)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    vocoder = Generator(config).to(device).eval()
    vocoder.load_state_dict(checkpoint_dict['generator'])
    vocoder.remove_weight_norm()

    return vocoder

def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values
    for batch in tqdm(batchs):
        # print("batch: ", batch)
        batch = to_device(batch, device)
        with torch.no_grad():
            # change ids to raw_texts
            # 0 for ids, 2 for speakers
            batch = list(batch)
            basenames = batch[0]
            # batch[0] = batch[1]
            batch = tuple(batch)
            # speaker = batch[2].tolist()
            speaker = batch[2]
            tmpname = [str(speaker[i]) + "-" + "mel" + "-" + basenames[i] + ".npy" for i in range(len(basenames))]
            os.makedirs((os.path.join("/workspace/nartts/AdaSpeech/output/en_cn_ouput_mel/")), exist_ok=True)
            count = 0
            for i in range(len(speaker)):
                # print("Saving \"{}\"...".format(tmpname))
                if not os.path.exists(os.path.join("/workspace/nartts/AdaSpeech/output/en_cn_ouput_mel/", tmpname[i])):
                    if count == 0:
                        # Forward
                        output = model.inference(
                            *(batch),
                            p_control=pitch_control,
                            e_control=energy_control,
                            d_control=duration_control,
                        )
                        count += 0
                        # --only mel spectrum
                    mel_spectrogram = output[1].transpose(1, 2)
                    np.save(
                            os.path.join("/workspace/nartts/AdaSpeech/output/en_cn_ouput_mel/", tmpname[i]),
                            mel_spectrogram[i].cpu().numpy(),
                        )

def synthesize_batch(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values
    for batch in tqdm(batchs):
        # print("batch: ", batch)
        batch = to_device(batch, device)
        with torch.no_grad():
            # change ids to raw_texts
            # 0 for ids, 2 for speakers
            batch = list(batch)
            basenames = batch[0]
            # batch[0] = batch[1]
            batch = tuple(batch)
            # speaker = batch[2].tolist()
            speaker = batch[2]
            mel_name = [str(speaker[i]) + "-" + "mel" + "-" + basenames[i] + ".npy" for i in range(len(basenames))]
            mel_save_path = "/data/speech_data/cuijiayan/vocoder_dataset/libri/mels/"
            father_dir = "/workspace/nartts/AdaSpeech/preprocessed_data/libri_spkr/"

            os.makedirs((os.path.join(mel_save_path)), exist_ok=True)
            count = 0
            duration_name = mel_name.replace('mel', 'duration')
            energy_name = mel_name.replace('mel', 'energy')
            pitch_name = mel_name.replace('mel', 'pitch')
            spk_name = mel_name.replace('mel', 'spk')
            spk = np.load(os.path.join(father_dir, "spk_emb", duration_name))
            for i in range(len(speaker)):
                # print("Saving \"{}\"...".format(tmpname))
                if not os.path.exists(os.path.join(mel_save_path, mel_name[i])):
                    if count == 0:
                        # Forward
                        output = model.inference(
                            *(batch),
                            p_targets = np.load(os.path.join(father_dir, "pitch", pitch_name)),
                            e_targets = np.load(os.path.join(father_dir, "energy", energy_name)),
                            d_targets = np.load(os.path.join(father_dir, "duration", duration_name)),
                            p_control=pitch_control,
                            e_control=energy_control,
                            d_control=duration_control,
                        )
                        count += 0
                        # --only mel spectrum
                    mel_spectrogram = output[1].transpose(1, 2)
                    np.save(
                            os.path.join(mel_save_path, mel_name[i]),
                            mel_spectrogram[i].cpu().numpy(),
                        )

            
def get_reference_mel(reference_audio_dir, STFT):
    wav, _ = librosa.load(reference_audio_dir)
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, STFT)
    return mel_spectrogram

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--reference_audio",
        type=str
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        default="single",
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )

    parser.add_argument(
        "--speaker_id",
        type=str,
        default=0,
        help="speaker ID(key: str) for multi-speaker synthesis, for single-sentence mode only",
    )

    parser.add_argument(
        "--language_id",
        type=int,
        default=0,
        help="language ID for multi-language synthesis"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
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

    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None
    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader = yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = preprocess_config, model_config, train_config

    STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )
    # speakers is str in speakers.json
    preprocess_config["path"]["datasets_base_path"]
    datasets = preprocess_config["dataset"]
    speakers_dic = {}
    max_value = 0
    for i, d in enumerate(datasets):
        with open(os.path.join(preprocess_config["path"]["datasets_base_path"], d, "speakers.json")) as f:
            tem = json.load(f)
            for dic in tem:
                dic_new = dic + '_' + d
                if dic_new not in speakers_dic:
                    if max_value <= tem[dic]:
                        max_value = tem[dic]
                        speakers_dic[dic_new] = tem[dic]
                    else:
                        speakers_dic[dic_new] = max_value + 1
                        max_value = speakers_dic[dic_new]

    # print("mel-inference ======== speakers_dic", speakers_dic)
    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    # vocoder = get_vocoder(args.vocoder_config, args.vocoder_checkpoint)
    vocoder = 0

    wav_path = args.reference_audio
    languages = np.array([args.language_id])
    # mel_spectrogram = get_reference_mel(wav_path, STFT)
    # mel_spectrogram = np.array([mel_spectrogram])


    mel_spectrogram = np.array([0, 0])
    control_values = args.pitch_control, args.energy_control, args.duration_control

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=dataset.collate_fn,
        )
        synthesize_batch(model, args.restore_step, configs, vocoder, batchs, control_values)
    
    if args.mode == "single":
        # write the new speaker in speakers.json
        ids = [wav_path.split('/')[-1].split('.')[0].split('_', 1)[1]]
        raw_texts = [args.text[:100]]
        # assert 0 == 1 
        if args.speaker_id in speakers_dic:
            speakers = np.array([speakers_dic[args.speaker_id]])
        else:
            speakers_dic[args.speaker_id] = speakers_dic[list(speakers_dic.keys())[-1]]+ 1
            speakers = np.array([speakers_dic[args.speaker_id]])
            with open(os.path.join(preprocess_config["path"]["datasets_base_path"], preprocess_config["dataset"][0], "speakers_combine.json"), "w") as f:
                json.dump(speakers_dic, f)

        # speakers = np.array([speakers_dic[args.speaker_id]])
        # if preprocess_config["preprocessing"]["text"]["language"] == "en":
        #     texts = np.array([preprocess_english(args.text, preprocess_config)])
        # elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
        #     texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        texts = np.array([preprocess_en_cn(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])

        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens), mel_spectrogram, languages)]

        speaker, id = wav_path.split('/')[-1].split('.')[0].split('_', 1)
        if len(preprocess_config["dataset"]) >1 :
            embedding_path = os.path.join(preprocess_config["path"]["datasets_base_path"], preprocess_config["dataset"][int(languages)], "speaker_embed",f"{speaker}-spk-{id}.npy")
        else:
            embedding_path = os.path.join(preprocess_config["path"]["datasets_base_path"], preprocess_config["dataset"][0], "speaker_embed",f"{speaker}-spk-{id}.npy")
        # print("speaker: ", speaker, "id: ", id)
        # print("wav_path: ", wav_path)
        # print("embedding_path: ", embedding_path)
        
        ## check embedding exists
        if os.path.exists(embedding_path):
            pass
        else:
            print("inference.py", embedding_path, "doesn't exist! Regenerated! Check it!")
            model_spkr = Model.from_pretrained("pyannote/embedding",use_auth_token="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")
            spkr_embedding = Inference(model_spkr, window="whole")
            audio, sampling_rate = sf.read(os.path.join(wav_path))
            # print("audio: ", audio.shape, "sampling_rate: ", sampling_rate)
            audio = normalize(audio) * 0.95
            emb = spkr_embedding({'waveform': torch.FloatTensor(audio).unsqueeze(0), 'sample_rate': sampling_rate})
            emb_tensor = torch.from_numpy(emb).float()
            os.makedirs(os.path.dirname(embedding_path), exist_ok=True) 
            torch.save(emb_tensor, embedding_path)
            print("save embedding succeed! : ", embedding_path)
            assert os.path.exists(embedding_path)


        # Synthesize
        synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)
