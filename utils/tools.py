import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt
import re
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.use("Agg")
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def to_device(data, device):
    if len(data) != 8:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
            avg_mel_phs,
            spk_embs,
            lang_ids,
        ) = data

        # speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)
        avg_mel_phs = torch.from_numpy(avg_mel_phs).float().to(device)
        spk_embs = torch.from_numpy(spk_embs).float().to(device)
        lang_ids = torch.from_numpy(lang_ids).long().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
            avg_mel_phs,
            spk_embs,
            lang_ids
        )

    if len(data) == 8:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len, mel, lang_ids) = data
        # speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mel = torch.from_numpy(mel).float().to(device)
        mel = torch.transpose(mel, 1, 2)
        languages = torch.from_numpy(lang_ids).long().to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len, mel, languages)


def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)
        logger.add_scalar("Loss/phone_level_loss", losses[6], step)
        logger.add_scalar("Loss/disc_loss", losses[7], step)
        logger.add_scalar("Loss/adv_loss", losses[8], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)
# batch = 
#   0           1        2      3          4              5      6         7             8         9         10       11           12         13           14
# ids, raw_texts, speakers, texts, text_lens, max(text_lens), mels, mel_lens, max(mel_lens), pitches, energies, durations, avg_mel_phs, spk_embs, language_ids
def infer_mels(targets, predictions, vocoder, model_config, preprocess_config):
    mels = []
    wavs = []
    fake_wavs = []
    tags = []
    for i in range(len(targets[0])):
        basename = targets[0][i]
        speakernames = targets[2][i]
        src_len = predictions[9][i].item()
        mel_len = predictions[10][i].item()
        mel_target = targets[6][i, :mel_len].detach().transpose(0, 1)
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = targets[11][i, :src_len].detach().cpu().numpy()

        if vocoder is not None:
            from .model import vocoder_infer

            wav_reconstruction = vocoder_infer(
                mel_target.unsqueeze(0),
                vocoder,
                model_config,
                preprocess_config,
            )[0]
            wav_prediction = vocoder_infer(
                mel_prediction.unsqueeze(0),
                vocoder,
                model_config,
                preprocess_config,
            )[0]
        else:
            wav_reconstruction = wav_prediction = None
        mels.append(mel_prediction.unsqueeze(0))
        wavs.append(wav_reconstruction)
        fake_wavs.append(wav_prediction)
        tags.append(speakernames+"_"+basename)

    return mels, wavs, fake_wavs, tags


def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):
    basename = targets[0][0]
    speakernames = targets[2][0]
    src_len = predictions[9][0].item()
    mel_len = predictions[10][0].item()
    mel_target = targets[6][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    duration = targets[11][0, :src_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = targets[9][0, :src_len].detach().cpu().numpy()
        pitch = expand(pitch, duration)
    else:
        pitch = targets[9][0, :mel_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = targets[10][0, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = targets[10][0, :mel_len].detach().cpu().numpy()

    if len(preprocess_config["path"]["combined_path"]) != 0:
        with open(os.path.join(preprocess_config["path"]["combined_path"], "stats_combined.json"), "r") as f:
            stats = json.load(f)
            print("tools.py 144 :  read right stats.json")
    else:
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json"), "r") as f:
            stats = json.load(f)  
            print("tools.py 147 :  read wrong stats.json")
    stats = stats["pitch"] + stats["energy"][:2]
    # print("mel_target: ", mel_target)
    # print("torch.max(mel_target)",torch.max(mel_target))
    # print("torch.min(mel_target)",torch.min(mel_target))
    # print("torch.mean(mel_target)",torch.mean(mel_target))
    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, speakernames+"_"+basename

def check_and_rename_file(path, filename):
    basename, ext = os.path.splitext(filename)
    basename = re.sub(r'([\^\*&%#@.,\-?!;:"\'()<>[\]{}…=+~\\$|/]|[\s]|[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5])+$', '', basename)
    basename = re.sub(r'[\^\*&%#@.,\-?!;:"\'()<>[\]{}…=+~\\$|/]|[\s]|[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]', '_', basename)
    basename = re.sub(r'[\_]+', '_', basename)
    basename = basename.lstrip('_')
    basename = basename[:60]
    file_exists = os.path.exists(os.path.join(path, basename+ext))
    if file_exists:
        # Extract the basename and extension from the filename
        i = 1
        new_filename = "{}_{}{}".format(basename, i, ext)
        
        while os.path.exists(os.path.join(path, new_filename)):
            i += 1
            new_filename = "{}_{}{}".format(basename, i, ext)
        return new_filename
    return os.path.join(path, basename+ext)

def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path):

    basenames = targets[1]
    speaker = targets[2]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        speaker = speaker[i]
        src_len = predictions[9][i].item()
        mel_len = predictions[10][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[6][i, :src_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions[2][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions[3][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[3][i, :mel_len].detach().cpu().numpy()

        with open(
            os.path.join(preprocess_config["path"]["combined_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        pic_path = os.path.join(path, "mel")
        if not os.path.exists(pic_path):
            os.makedirs(pic_path)
        tmpname = check_and_rename_file(pic_path, "{}_{}.png".format(speaker, basename))
        figname = os.path.join(pic_path, tmpname)
        print("Saving \"{}\"...".format(figname))
        plt.savefig(os.path.join(pic_path, figname))
        plt.close()
    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[10] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        tmpname = check_and_rename_file(path, "{}_{}.wav".format(speaker, basename))
        wavname = os.path.join(path, tmpname)
        wavfile.write(os.path.join(wavname), sampling_rate, wav)
        print("Saving wav \"{}\"...".format(wavname))
        
    return mel_predictions, wavname.split('.')[0].split("/")[-1]

def synth_samples_wav(targets, predictions, vocoder, model_config, preprocess_config, path):
    """
    Synthesize audio files from batch predictions
    Args:
        targets: input data containing text and speaker information
        predictions: model output predictions
        vocoder: vocoder model for waveform generation
        model_config: model configuration
        preprocess_config: preprocessing configuration
        path: path to save generated audio files
    Returns:
        wav_paths: list of paths to generated audio files
    """
    basenames = targets[1]
    speaker = targets[2]
    wav_paths = []

    # Create output directory if not exists
    os.makedirs(path, exist_ok=True)

    # Generate mel spectrograms and convert to audio
    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[10] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    from .model import vocoder_infer
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    # Save each audio file
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        # Generate unique filename
        # tmpname = f"{speaker}_{basename}.wav"
        wavname = check_and_rename_file(path, f"{speaker}_{basename}.wav")
        os.makedirs(os.path.dirname(wavname), exist_ok=True)
        # Save audio file
        wavfile.write(wavname, sampling_rate, wav)
        print(f'Saving wav "{wavname}"...')
        wav_paths.append(wavname)
    
    return wav_paths


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
