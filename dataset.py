import json
import math
from operator import sub
import os
import audio as Audio
import numpy as np
from torch.utils.data import Dataset
import librosa
from text import text_to_sequence
from utils.tools import pad_1D, pad_2D
import yaml

np.random.seed(1234)
class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.cleaners = []
        self.dataset_name = preprocess_config["dataset"]
        self.datasets_base_path = preprocess_config["path"]["datasets_base_path"]
        self.preprocessed_path = []
        for d in self.dataset_name:
            self.preprocessed_path.append(os.path.join(self.datasets_base_path, d))
            
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.spk_type = train_config["spk-type"]
        self.basename, self.speaker, self.text, self.raw_text, self.lang_id = self.process_meta(
            filename
        )
        self.speaker_map = {}
        max_value = 0
        for i, p in enumerate(self.preprocessed_path):
            with open(os.path.join(p, "speakers.json")) as f:
                tem = json.load(f)
                for dic in tem:
                    dic_new = dic + '_' + d
                    if dic_new not in self.speaker_map:
                        if max_value <= tem[dic]:
                            max_value = tem[dic]                            
                            self.speaker_map[dic_new] = tem[dic]
                        else:
                            self.speaker_map[dic_new] = max_value + 1
                            max_value = self.speaker_map[dic_new]
        if len(preprocess_config["path"]["combined_path"]) != 0:
            with open(os.path.join(preprocess_config["path"]["combined_path"], "stats_combined.json"), "r") as f:
                stats = json.load(f)
                print("dataset.py 48 :  read right stats.json") 
        else:
            with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json"), "r") as f:
                stats = json.load(f)     
                print("dataset.py 51 :  read wrong stats.json")   
        self.pitch = stats["pitch"]
        self.energy = stats["energy"]
        self.sort = sort
        self.drop_last = drop_last 
        self.count = 0

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        raw_text = self.raw_text[idx]
        language_id = self.lang_id[idx]
        text = self.text[idx]
        # speaker_id = self.speaker_map[speaker]

        if language_id == 0:  # en libri 
            phone = np.array(text_to_sequence(text, self.cleaners))
            p_path = self.preprocessed_path[0]
        elif language_id == 1:  # cn cctv
            phone = np.array(text_to_sequence(text, ""))
            p_path = self.preprocessed_path[1] if len(self.preprocessed_path) > 1 else self.preprocessed_path[0]
        elif language_id == 2:  # cn aishell
            phone = np.array(text_to_sequence(text, ""))
            p_path = self.preprocessed_path[2] if len(self.preprocessed_path) > 2 else self.preprocessed_path[0]

        mel_path = os.path.join(p_path, "mel", "{}-mel-{}.npy".format(speaker, basename))
        mel = np.load(mel_path)
    
        # continue loading data
        pitch_path = os.path.join(
            p_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        pitch = (pitch - self.pitch[2])/ self.pitch[3]
        energy_path = os.path.join(
            p_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        energy = (energy - self.energy[2])/ self.energy[3]

        duration_path = os.path.join(
            p_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)
        avg_mel_ph_path = os.path.join(
            p_path,
            "avg_mel_phon",
            "{}-avg_mel-{}.npy".format(speaker, basename),
        )
        avg_mel_ph = np.load(avg_mel_ph_path)

        if self.spk_type == "spk":
            spk_emb_path = os.path.join(
                p_path,
                "spk",
                "{}-spk-{}.npy".format(speaker, basename),
            )
            spk_emb = np.load(spk_emb_path)

        if self.spk_type == "xlsr":
            xlsr_path = os.path.join(
                p_path,
                "xlsr",
                "{}-xlsr-{}.npy".format(speaker, basename),
            )
            spk_emb = np.load(xlsr_path)
            spk_emb = np.squeeze(spk_emb, axis=0)
        sample = {
            "id": basename,
            "speaker": speaker,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "avg_mel_ph": avg_mel_ph,
            "spk_emb": spk_emb,
            "language_id": language_id
        }
        return sample

    def process_meta(self, filename):
        name = []
        speaker = []
        text = []
        raw_text = []
        lang_id = []

        for p in self.preprocessed_path:
            with open(
                os.path.join(p, filename), "r", encoding="utf-8"
            ) as f:
                for line in f.readlines():
                    n, s, t, r, l = line.strip("\n").split("|")
                    name.append(n)
                    speaker.append(s)
                    text.append(t) # phoneme
                    raw_text.append(r) # raw text
                    lang_id.append(int(l))
        return name, speaker, text, raw_text, lang_id

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        # print("dataset.py====================reprocess ids:", ids)
        speakers = [data[idx]["speaker"] for idx in idxs]
        # print("dataset.py====================reprocess speakers:", speakers)
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        avg_mel_phs = [data[idx]["avg_mel_ph"] for idx in idxs]
        spk_embs = [data[idx]["spk_emb"] for idx in idxs]
        language_ids = [data[idx]["language_id"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        avg_mel_phs = pad_2D(avg_mel_phs)
        spk_embs = pad_1D(spk_embs)
        language_ids = np.array(language_ids)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
            avg_mel_phs,
            spk_embs,
            language_ids
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            # print("dataset.py====================collate_fn idx:", idx)
            output.append(self.reprocess(data, idx))
        return output

def get_reference_mel(reference_audio_dir, STFT):
    wav, _ = librosa.load(reference_audio_dir)
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, STFT)
    return mel_spectrogram
class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.dataset_name = preprocess_config["dataset"]
        self.datasets_base_path = preprocess_config["path"]["datasets_base_path"]
        self.preprocessed_path = []
        for d in self.dataset_name:
            self.preprocessed_path.append(os.path.join(self.datasets_base_path, d))

        max_value = 0
        self.speaker_map = {}
        max_value = 0
        for i, p in enumerate(self.preprocessed_path):
            with open(os.path.join(p, "speakers.json")) as f:
                tem = json.load(f)
                for dic in tem:
                    dic_new = dic + '_' + self.dataset_name[i]
                    if dic_new not in self.speaker_map:
                        if max_value <= tem[dic]:
                            max_value = tem[dic]                            
                            self.speaker_map[dic_new] = tem[dic]
                        else:
                            self.speaker_map[dic_new] = max_value + 1
                            max_value = self.speaker_map[dic_new]
        # print("self.speaker_map: ", self.speaker_map)
        self.basename, self.speaker, self.text, self.raw_text, self.lang_id = self.process_meta(
            filepath
        )
        self.STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        # speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        language = self.lang_id[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        # print("dataset ===== ",basename, speaker_id, phone, raw_text)
        return (basename, speaker, phone, raw_text, language)

    def process_meta(self, filenames):
        name = []
        speaker = []
        text = []
        raw_text = []
        language = []
        with open(filenames, "r", encoding="utf-8") as f:
            for line in f.readlines():
                a = line.strip("\n").split("|")
                name.append(a[0])
                speaker.append(a[1]+'_'+self.dataset_name[int(a[4])])
                text.append(a[2])
                raw_text.append(a[3])
                language.append(a[4])
        return name, speaker, text, raw_text, language

    def collate_fn(self, data):
        # mel_spectrogram = np.array([0, 0, 0])
        # mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
        # mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        language = np.array([int(d[4]) for d in data])
        text_lens = np.array([text.shape[0] for text in texts])
        mel_spectrograms = [get_reference_mel(os.path.join("/data/speech_data/LibriTTS/audios/en",speakers[i].split("_",1)[0]+'_'+ids[i]+".wav"), self.STFT) for i in range(len(ids))]
        # mel_spectrograms = np.array([get_reference_mel(os.path.join("/data/speech_data/LibriTTS/audios/en",speakers[i].split("_",1)[0]+'_'+ids[i]+".wav"), self.STFT) for i in range(len(ids))])
        mel_len = [m.shape[1] for m in mel_spectrograms]
        max_len = max(mel_len)
        mel_spectrograms = pad_2D(np.array([
                get_reference_mel(os.path.join("/data/speech_data/LibriTTS/audios/en",speakers[i].split("_",1)[0]+'_'+ids[i]+".wav"), self.STFT) for i in range(len(ids))]), max_len)
        texts = pad_1D(texts)
        return ids, raw_texts, speakers, texts, text_lens, max(text_lens), mel_spectrograms, language



