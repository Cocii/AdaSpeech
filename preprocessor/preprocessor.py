import os
import random
import json

import tgt
import torch
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from tqdm import tqdm

import audio as Audio
def find_wav_files(directory):
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files

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

class Preprocessor:
    def __init__(self, config):
        self.config = config
        # textgrid in main_path
        self.main_path = config["path"]["main_path"]
        # .lab .wav in self.in_dir
        self.in_dir = config["path"]["raw_path"]
        # output dir
        self.out_dir = config["path"]["preprocessed_path"]

        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "avg_mel_phon")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        num_spk = 0
        
        # Put all audio files into a single folder and differentiate speakers by names
        for id_language, language in enumerate(tqdm(os.listdir(self.in_dir))):
            for wav in tqdm(os.listdir(os.path.join(self.in_dir, language))):
                speaker, wav_name = wav.split("_", 1)
                if speaker not in speakers:
                    speakers[speaker] = num_spk
                    num_spk += 1
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.replace(".wav","")
                tg_path = os.path.join(
                    self.main_path, "TextGrid", speaker + "_{}.TextGrid".format(basename)
                )
                pitch = np.array([])
                energy = np.array([])
                if os.path.exists(tg_path):
                    ret = self.process_utterance(language, id_language, speaker, basename)
                    if ret is None:
                        continue
                    else:
                        info, pitch, energy, n = ret
                    out.append(info)    
                # print("len(pitch): ", len(pitch))
                # print("len(energy): ", len(energy))
                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

                n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min = 0
        pitch_max = 0
        energy_min = 0
        energy_max = 0
        
        # pitch_min, pitch_max = self.normalize(
        #     os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        # )
        # energy_min, energy_max = self.normalize(
        #     os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        # )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self,language, language_id, speaker, basename):
        wav_path = os.path.join(self.in_dir, language, speaker + "_{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, language, speaker + "_{}.lab".format(basename))
        tg_path = os.path.join(
            self.main_path, "TextGrid", speaker + "_{}.TextGrid".format(basename)
        )
        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)

        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]
        
        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]
        
        duration = torch.from_numpy(np.array(duration))
        avg_mel_ph = self.average_mel_by_duration(mel_spectrogram, duration)
        assert (avg_mel_ph.shape[0] == duration.shape[-1])

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        avg_mel_filename = "{}-avg_mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "avg_mel_phon", avg_mel_filename),
            avg_mel_ph.numpy()
        )
        return (
            "|".join([basename, speaker, text, raw_text, str(language_id)]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            # Trim leading silences
            if p == "":
                p = "sp"

            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]
        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    # def normalize(self, in_dir, mean, std):
    #     max_value = np.finfo(np.float64).min
    #     min_value = np.finfo(np.float64).max
    #     for filename in os.listdir(in_dir):
    #         filename = os.path.join(in_dir, filename)
    #         values = (np.load(filename) - mean) / std
    #         np.save(filename, values)

    #         max_value = max(max_value, max(values))
    #         min_value = min(min_value, min(values))

    #     return min_value, max_value

    # def average_mel_by_duration(self, mel, duration):
    #     if duration.sum() != mel.shape[-1]:
    #         duration[-1] += 1
    #     d_cumsum = F.pad(duration.cumsum(dim=0), (1, 0))
    #     x_avg = [
    #             np.sum(mel[:, int(start):int(end)], axis=1)//np.array(end - start) if len(mel[:, int(start):int(end)]) != 0 else mel.zeros()
    #             for start, end in zip(d_cumsum[:-1], d_cumsum[1:])
    #         ]
    #     x_avg = torch.from_numpy(np.array(x_avg))
    #     return x_avg

    def average_mel_by_duration(self, mel, duration):
        mel = torch.from_numpy(mel)
 
        if duration.sum() != mel.shape[-1]:
            duration[-1] += 1
        d_cumsum = F.pad(duration.cumsum(dim=0), (1, 0))
        x_avg = [
            torch.sum(
                mel[:, int(start):int(end)], dim=1) / 
                torch.clamp((end - start), min=1) 
                if end > start else torch.zeros_like(mel[:, 0])
            for start, end in zip(d_cumsum[:-1], d_cumsum[1:])
        ]
        x_avg = torch.stack(x_avg)
        return x_avg


