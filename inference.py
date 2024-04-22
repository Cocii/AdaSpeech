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
from utils.tools import to_device, synth_samples, AttrDict, check_and_rename_file
from torch.utils.data import DataLoader
from dataset import Dataset
from text import text_to_sequence
from text.cleaners import english_cleaners
from datetime import datetime
from g2p_en import G2p
from string import punctuation
import audio as Audio
import os
import soundfile as sf
from librosa.util import normalize
# sys.path.append("vocoder")
import re
from xlsr.wav2vec2_speaker_encoder import SpeakerEncoder

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
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"], "english_cjy.dict")

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

def preprocess_en_cn(text, preprocess_config):
    text = re.sub(r'([\^\*&%#@.,\-?!;:"\'()<>[\]{}…=+~\\$|/]|[\s]|[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5])+$', '', text)
    lexicon = read_lexicon(os.path.join(preprocess_config["path"]["lexicon_path"], "english_cjy.dict"))
    lexicon_pinyin = read_lexicon(os.path.join(preprocess_config["path"]["lexicon_path"], "pinyin-lexicon-r.dict"))
    g2p = G2p()
    phones = []
    words = re.split(r'([\^\*&%#@.,\-?!;:"\'()<>[\]{}…=+~\\$|/]|[\s]|[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]|[\u4e00-\u9fa5])', text)
    # 中文 [\u4e00-\u9fa5]
    # 中文标点 [\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]
    # print("\nwords : ", words)
    for w in words:
        if w is not None:
            if is_chinese_char(w):
                p_pinyin = pinyin(w, style=Style.TONE3, strict=False, neutral_tone_with_five=True)
                p_pinyin = p_pinyin[0]
                p_pinyin = p_pinyin[0]
                if p_pinyin in lexicon_pinyin:
                    phones += lexicon_pinyin[p_pinyin]
                else:
                    phones.append("sp")
            else:
                tmp_cleaned = re.split(r'[\-_\s]',english_cleaners(w))
                tem_cleaned = [tmp for tmp in tmp_cleaned if tmp != ""]
                # print("tmp_cleaned: ", tem_cleaned)
                for tmp in tmp_cleaned:
                    if tmp.lower() in lexicon:
                        phones += lexicon[tmp.lower()]
                    else:
                        phones += list(filter(lambda p: p != " ", g2p(tmp.lower())))

    # print("\nwords : ", phones)

    phones = "{" + "}{".join(phones) + "}"
    # print("phones before: ", phones)
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    # print("phones before: ", phones)
    # for p in phones:

    phones = phones.replace("}{", " ")
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, []
        )
    )
    # print("sequence: ", sequence)
    return np.array(sequence)

    
def is_chinese_char(char):
    # including punctuation
    patten = r"[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]"
    if char is None or len(char) == 0 or len(char) > 1:
        return False
    if 0x4e00 <= ord(char) <= 0x9fff or re.search(patten,char):
        return True
    else:
        return False
    
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
    vocoder_name = checkpoint_path.split("/")[-2]
    bigvgan_name = checkpoint_path.split("/")[-3]
    bigvgan_name1 = checkpoint_path.split("/")[-4]
    if vocoder_name == "hifigan":
        from vocoder.models.hifigan import Generator
    elif vocoder_name == "BigVGAN" or vocoder_name == "bigvgan_22khz_80band" or bigvgan_name == "bigvgan_22khz_80band" or bigvgan_name1 ==  "bigvgan_22khz_80band":
        from vocoder.models.BigVGAN import BigVGAN as Generator
    else:
        print("error in vocoder loading process! check it!  fintune.py 26")

    config = json.load(open(config, 'r', encoding='utf-8'))
    config = AttrDict(config)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    vocoder = Generator(config).to(device).eval()
    vocoder.load_state_dict(checkpoint_dict['generator'])
    vocoder.remove_weight_norm()
    return vocoder

def synthesize(model, step, configs, vocoder, batchs, control_values, ref_path = None):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values
    emb_np = np.load(ref_path)
    emb_torch = torch.from_numpy(emb_np).float().to(device)
    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model.inference(
                *(batch),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
                spk_emb = emb_torch
            )

            # change ids to raw_texts
            # 0 for ids, 2 for speakers
            
            batch = list(batch)
            batch = [[i] for i in batch]
            batch[0] = batch[1]
            batch = tuple(batch)
            mel_spectrogram, tmpname = synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )
            # saving mel
            os.makedirs((os.path.join(train_config["path"]["result_path"], "mel")), exist_ok=True)
            tmpname = tmpname + ".npy"
            print("Saving \"{}\"...".format(os.path.join(train_config["path"]["result_path"], "mel", tmpname)))
            np.save(
                    os.path.join(train_config["path"]["result_path"], "mel", tmpname),
                    mel_spectrogram.cpu().numpy(),
                )
            
def get_reference_mel(reference_audio_dir, STFT):
    wav, _ = librosa.load(reference_audio_dir)
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, STFT)
    return mel_spectrogram

def generate_xlsr(wav_path, save_path, spk_encoder):
    # generate
    # speaker, id = wav_path.split('/')[-1].split('.')[0].split('_', 1)
    data, _ = librosa.load(wav_path, sr=16000)
    data = np.expand_dims(data, axis=0)
    x = torch.from_numpy(data)
    x = x.to(device)
    a = spk_encoder(x).detach().cpu().numpy()
    os.makedirs(os.path.dirname(save_path), exist_ok=True) 
    np.save(save_path, a)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--reference_audio",
        type=str,
        default=None
    )
    parser.add_argument(
        "--reference_embedding",
        type=str,
        default=None
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
    spk_type = model_config["spk-type"]

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

    # print("inference.py ====== speakers_dic", speakers_dic)
    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(args.vocoder_config, args.vocoder_checkpoint)
    wav_path = args.reference_audio
    languages = np.array([args.language_id])
    mel_spectrogram = get_reference_mel(wav_path, STFT)
    mel_spectrogram = np.array([mel_spectrogram])
    # if preprocess_config["dataset"][int(languages)] == "aishell3":
    if args.language_id == 2:
        speaker = wav_path.split('/')[-1].split('.')[0][:7]
        id = wav_path.split('/')[-1].split('.')[0]
    else:
        speaker, id = wav_path.split('/')[-1].split('.')[0].split('_', 1)
    
    if len(speaker) < 4:
        speaker = speaker.zfill(4)

    if args.reference_embedding != None:
        embedding_path = args.reference_embedding
    else:
        if len(preprocess_config["dataset"]) >1:
            if args.language_id == 2:
                embedding_path = os.path.join("/workspace/nartts/AdaSpeech/preprocessed_data/aishell3", spk_type,f"{speaker}-" + spk_type + f"-{id}.npy")
            else:
                embedding_path = os.path.join(preprocess_config["path"]["datasets_base_path"], preprocess_config["dataset"][int(languages)], spk_type,f"{speaker}-" + spk_type + f"-{id}.npy")
        else:
            # if preprocess_config["dataset"][0] == "xjp_data":
            #     embedding_path = os.path.join(preprocess_config["path"]["datasets_base_path"], preprocess_config["dataset"][0], "speaker_embed","wav-spk-xjp_{id}.npy")
            embedding_path = os.path.join(preprocess_config["path"]["datasets_base_path"], preprocess_config["dataset"][0], spk_type,f"{speaker}-" + spk_type + f"-{id}.npy")
    
    ## check embedding exists
    # assert os.path.exists(embedding_path)
    if os.path.exists(embedding_path):
        print("Exist embedding_path", embedding_path)
    else:
        print("inference.py 389", embedding_path, "doesn't exist! Regenerated! Check it!")

        if spk_type == "spk":
            from pyannote.audio import Inference
            from pyannote.audio import Model
            model_spkr = Model.from_pretrained("pyannote/embedding",use_auth_token="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")
            spkr_embedding = Inference(model_spkr, window="whole")
            audio, sampling_rate = librosa.load(os.path.join(wav_path), sr=22050)
            audio = normalize(audio) * 0.95
            emb = spkr_embedding({'waveform': torch.FloatTensor(audio).unsqueeze(0), 'sample_rate': sampling_rate})
            np.save(embedding_path, emb)
        if spk_type == 'xlsr':
            spk_encoder = SpeakerEncoder().to(device)
            generate_xlsr(wav_path, embedding_path, spk_encoder)

        print("save embedding succeed! : ", embedding_path)
        assert os.path.exists(embedding_path)
        
    control_values = args.pitch_control, args.energy_control, args.duration_control
    batchs = []
    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=dataset.collate_fn,
        )
        batchs[6] = [mel_spectrogram for _ in batchs[6]]
        synthesize(model, args.restore_step, configs, vocoder, batchs, control_values, embedding_path)


    if args.mode == "single":
        # write the new speaker in speakers.json
        raw_texts = []
        with open('/workspace/AdaSpeech/test.txt', 'r') as file:
            for line in file:
                raw_texts.append(line.strip())
        # assert 0 == 1 
        speakers = [args.speaker_id]
        # if args.speaker_id in speakers_dic:
        #     speakers = np.array([speakers_dic[args.speaker_id]])
        # else:
        #     speakers_dic[args.speaker_id] = speakers_dic[list(speakers_dic.keys())[-1]]+ 1
        #     speakers = np.array([speakers_dic[args.speaker_id]])
        #     with open(os.path.join(preprocess_config["path"]["datasets_base_path"], preprocess_config["dataset"][0], "speakers_combine.json"), "w") as f:
        #         json.dump(speakers_dic, f)

        # speakers = np.array([speakers_dic[args.speaker_id]])
        # if preprocess_config["preprocessing"]["text"]["language"] == "en":
        #     texts = np.array([preprocess_english(args.text, preprocess_config)])
        # elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
        #     texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        texts = [np.array([preprocess_en_cn(i, preprocess_config)]) for i in raw_texts if i != '']
        text_lens = [np.array([len(i)]) for i in texts]
        for i in range(len(texts)):
            batchs = [(id, raw_texts[i], speakers, texts[i], text_lens[i], max(text_lens[i]), mel_spectrogram, languages)]
            synthesize(model, args.restore_step, configs, vocoder, batchs, control_values, embedding_path)
 