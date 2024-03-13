import json
import os
import numpy as np
from tqdm import tqdm
import argparse
import yaml
from sklearn.preprocessing import StandardScaler
# attention!! calculate the min max value and normalize the energy pitch data  
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def normalize(in_dir, mean, std):
    max_value = np.finfo(np.float64).min
    min_value = np.finfo(np.float64).max
    for filename in os.listdir(in_dir):
        filename = os.path.join(in_dir, filename)
        values = (np.load(filename) - mean) / std
        np.save(filename, values)

        max_value = max(max_value, max(values))
        min_value = min(min_value, min(values))

    return min_value, max_value

def main(config):
    
    with open(os.path.join(config["path"]["preprocessed_path"], "stats.json"), "r") as f:
        stats = json.load(f)
    # stats["pitch"][2] # pitch mean
    # stats["pitch"][3] # pitch std 
    # stats["energy"][2] # energy mean
    # stats["energy"][3] # energy std
    pitch_path = os.path.join(config["path"]["preprocessed_path"], "pitch/")
    energy_path = os.path.join(config["path"]["preprocessed_path"], "energy/")
    max_value = np.finfo(np.float64).min
    min_value = np.finfo(np.float64).max
    print(pitch_path)
    print(energy_path)
    # assert pitch_path == energy_path

    # pitch_scaler = StandardScaler()
    # energy_scaler = StandardScaler()
    # pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
    # energy_normalization = config["preprocessing"]["energy"]["normalization"]
    # if pitch_normalization:
    #     pitch_mean = pitch_scaler.mean_[0]
    #     pitch_std = pitch_scaler.scale_[0]
    # else:
    #     # A numerical trick to avoid normalization...
    #     pitch_mean = 0
    #     pitch_std = 1
    # if energy_normalization:
    #     energy_mean = energy_scaler.mean_[0]
    #     energy_std = energy_scaler.scale_[0]
    # else:
    #     energy_mean = 0
    #     energy_std = 1



    for p in tqdm(os.listdir(pitch_path)):
        p = os.path.join(pitch_path, p)
        try:
            pitch = np.load(p)
            pitch = (pitch - stats["pitch"][2]) / stats["pitch"][3]
            max_value = max(max_value, max(pitch))
            min_value = min(min_value, min(pitch))
            np.save(p, pitch)
            stats["pitch"][0] = min_value
            stats["pitch"][1] = max_value
        except Exception as e:
            print("error p: ", p)

    max_value = np.finfo(np.float64).min
    min_value = np.finfo(np.float64).max

    for e in tqdm(os.listdir(energy_path)):
        e = os.path.join(energy_path, e)
        energy = np.load(e)
        energy = (energy - stats["energy"][2]) / stats["energy"][3]
        max_value = max(max_value, max(energy))
        min_value = min(min_value, min(energy))
        np.save(e, energy)
        stats["energy"][0] = min_value
        stats["energy"][1] = max_value

    with open(os.path.join(config["path"]["preprocessed_path"], "stats.json"), "w") as f:
        json.dump(stats, f, cls=MyEncoder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)