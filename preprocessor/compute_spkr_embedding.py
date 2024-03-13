import os
import soundfile as sf
import torch
from tqdm import tqdm
from librosa.util import normalize
import numpy as np
import argparse
import yaml





def main():
    # audios_dir = os.path.join("/data/speech_data/LibriTTS/audios", "en/")
    # emb_path = os.path.join("/workspace/nartts/AdaSpeech/preprocessed_data/libri_spkr/spk_emb/")

    audios_dir = os.path.join("/data/speech_data/cctv_cjy/")
    emb_path = os.path.join("/workspace/nartts/AdaSpeech/preprocessed_data/cctv_212018_yw/spk_emb/")

    from pyannote.audio import Inference, Model
    model = Model.from_pretrained("pyannote/embedding",use_auth_token="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")
    spkr_embedding = Inference(model, window="whole")
    
    i = 0

    k = 0
    for j in [2021,2020,2018]:
        for root, dirs, files in tqdm(os.walk(os.path.join(audios_dir, str(j)))):
            for file in files:
                if file.endswith(".lab"):
                    k += 1
    print("Total exist .lab: ", k)

    
    k = 0

    # total .lab: 355399
    # Total exist spkr_emb.npy:  81360
    # Total not exist spkr_emb.npy:  154332

    for j in [2021,2020,2018]:
        for root, dirs, files in tqdm(os.walk(os.path.join(audios_dir, str(j)))):
            for file in files:
                if file.endswith(".lab"):
                    file = file.split('.')[0]
                    file = file+'.wav'
                    audio_path = os.path.join(root, file)
                    speaker = audio_path.split('/')[-1].split('.')[0].split('_',1)[0]
                    id = audio_path.split('/')[-1].split('.')[0].split('_',1)[1]
                    embedding_path = os.path.join(emb_path, f"{str(j)}-spk-{speaker+'_'+id}.npy")
                    if os.path.exists(embedding_path):
                        i += 1
                    else:
                        audio_path = os.path.join(root, speaker+'_'+id+'.wav')
                        k+=1
                        audio, sampling_rate = sf.read(os.path.join(root, speaker+'_'+id+'.wav'))
                        audio = normalize(audio) * 0.95
                        if len(audio) > 22050:
                            # print("audio_path: ", audio_path, " ; audio len: ", len(audio))
                            emb = spkr_embedding({'waveform': torch.FloatTensor(audio).unsqueeze(0), 'sample_rate': sampling_rate})
                            os.makedirs(os.path.dirname(embedding_path), exist_ok=True) 
                            # with open("/workspace/nartts/AdaSpeech/preprocessor/error_files.txt", "a") as f:
                            #     f.write(audio_path + "\n" + embedding_path + '\n' + '\n')
                            np.save(embedding_path, emb)
                        
        print("Total exist spkr_emb.npy: ", i)
        print("Total not exist spkr_emb.npy: ", k)

#     for file in tqdm(os.listdir(audios_dir)):
#         if file.endswith('.wav'):
#             try:
#                 file = os.path.basename(file)
#                 speaker, remain = file.split('_', 1)
#                 id, extension = remain.split('.')
#                 embedding_path = os.path.join(emb_path, f"{speaker}-spk-{id}.npy")
#                 # print(embedding_path)
#                 if os.path.exists(embedding_path):
#                     i += 1
#                 else:
#                     audio, sampling_rate = sf.read(os.path.join(audios_dir, speaker+'_'+id+'.wav'))
#                     # print("audio: ", audio.shape, "sampling_rate: ", sampling_rate)
#                     audio = normalize(audio) * 0.95
#                     emb = spkr_embedding({'waveform': torch.FloatTensor(audio).unsqueeze(0), 'sample_rate': sampling_rate})
#                     os.makedirs(os.path.dirname(embedding_path), exist_ok=True) 
#                     # emb_tensor = torch.from_numpy(emb).float()
#                     # torch.save(emb_tensor, embedding_path)

#                     # save as numpy array
#                     np.save(emb, embedding_path)
#             except Exception as e:
#                 print("Error! :", e)
#                 error_files.append(file)
#     with open("error_files.txt", "w") as f:
#         for error_file in error_files:
#             f.write(error_file + "\n")
#     print("Total exist spkr_emb.npy: ", i)

if __name__ == "__main__":
    main()
    