import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

pitch_path = os.path.join("nartts/AdaSpeech/preprocessed_data/preprocessed_data_libri_spkr/pitch")
energy_path = os.path.join("nartts/AdaSpeech/preprocessed_data/preprocessed_data_libri_spkr/energy")
def process_file(p):
    try:
        pitch = np.load(p)
    except Exception as e:
        print("error p: ", p)

with ThreadPoolExecutor(max_workers=4) as executor: 
    file_paths = [os.path.join(pitch_path, p) for p in os.listdir(pitch_path)]
    
    # 使用tqdm包装，以显示进度条
    for _ in tqdm(executor.map(process_file, file_paths), total=len(file_paths)):
        pass
