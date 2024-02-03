import os
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import torch.nn as nn
import random
import torch
import matplotlib.pyplot as plt

pca = PCA(n_components=2)


num_points = 2000
aishell_path = "/workspace/nartts/AdaSpeech/preprocessed_data/aishell3"
libri_path = "/workspace/nartts/AdaSpeech/preprocessed_data/libri_spkr" 
cctv_path = "/workspace/nartts/AdaSpeech/preprocessed_data/cctv_212018_yw"

aishell_path = os.path.join(aishell_path, "spk_emb")
libri_path = os.path.join(libri_path, "spk_emb")
cctv_path = os.path.join(cctv_path, "spk_emb")
def list_files_in_directory(directory):
    file_paths = [entry.path for entry in os.scandir(directory) if entry.is_file()]
    return file_paths

def norm_all_audios(paths, num_points, spk_norm):
    spk = random.sample(list_files_in_directory(paths), num_points)
    spk_normalized = []
    for i in tqdm(spk):
        spk_normalized.append(spk_norm(torch.tensor(np.load(i), dtype=torch.float32)).detach().numpy())  
    return spk_normalized

spk_norm = nn.LayerNorm(512)
aishell_norm = norm_all_audios(aishell_path, num_points, spk_norm)
libri_norm = norm_all_audios(libri_path, num_points, spk_norm)
cctv_norm = norm_all_audios(cctv_path, num_points, spk_norm)
total_norm = aishell_norm + libri_norm + cctv_norm

pca.fit(total_norm)
reduced_data = pca.transform(total_norm)

plt.figure(figsize=(20, 12))
plt.plot(reduced_data[:num_points,0], reduced_data[:num_points,1], 'o', label='aishell', alpha = 0.5)
plt.plot(reduced_data[num_points:2*num_points,0], reduced_data[num_points:2*num_points,1], 's', label='libri', alpha = 0.5)
plt.plot(reduced_data[2*num_points:,0], reduced_data[2*num_points:,1], '*', label='cctv', alpha = 0.5)
plot_name = "PCA_" + str(num_points)
plt.title(plot_name)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
print("Saving picture:")
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
plt.savefig(os.path.join(parent_directory, "plots", plot_name+'.png'))
plt.show()