import os
import soundfile as sf
import torch
from tqdm import tqdm
from librosa.util import normalize
from pyannote.audio import Inference, Model
import librosa
import argparse
import yaml
import numpy as np

def main():
    # ===== audio IN/OUT =====
    # audios_dir = os.path.join("/data/speech_data/ref_audios/Trump_WEF_2018/0-98/", "en/")
    # emb_path = os.path.join("/workspace/nartts/AdaSpeech/preprocessed_data/trump/speaker_embed/")
    audios_dir = os.path.join("/data/speech_data/ref_audios/other")
    emb_path = os.path.join("/data/speech_data/ref_audios/other")
    

    model = Model.from_pretrained("pyannote/embedding",use_auth_token="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")
    spkr_embedding = Inference(model, window="whole")
    i = 0
    error_files = []
    for file in tqdm(os.listdir(audios_dir)):
        if file.endswith('.wav'):
            try:
                file = os.path.basename(file)
                speaker, remain = file.split('_', 1)
                id, extension = remain.split('.')
                embedding_path = os.path.join(emb_path, f"{speaker}-spk-{id}.npy")
                # print(embedding_path)
                if os.path.exists(embedding_path):
                    i += 1
                else:
                    audio, sampling_rate = librosa.load(os.path.join(audios_dir, speaker+'_'+id+'.wav'), sr=22050)
                    # print("audio: ", audio.shape, "sampling_rate: ", sampling_rate)
                    audio = normalize(audio) * 0.95
                    emb = spkr_embedding({'waveform': torch.FloatTensor(audio).unsqueeze(0), 'sample_rate': sampling_rate})

                    # =====save numpy===== 
                    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
                    np.save(embedding_path, emb)

                    # =====save torch=====
                    # emb_tensor = torch.from_numpy(emb).float()
                    # os.makedirs(os.path.dirname(embedding_path), exist_ok=True) 
                    # torch.save(emb_tensor, embedding_path)
            except Exception as e:
                print("Error! :", e)
                error_files.append(file)
    with open("error_files.txt", "w") as f:
        for error_file in error_files:
            f.write(error_file + "\n")
    print("Total exist spkr_emb.npy: ", i)

if __name__ == "__main__":
    main()
    