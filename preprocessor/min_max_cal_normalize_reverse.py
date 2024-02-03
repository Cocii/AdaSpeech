import json
import os
import numpy as np
from tqdm import tqdm
import argparse
import yaml
from sklearn.preprocessing import StandardScaler
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def main(config):
    
    with open(os.path.join(config["path"]["preprocessed_path"], "stats.json"), "r") as f:
        stats = json.load(f)
    # stats["pitch"][2] # pitch mean
    # stats["pitch"][3] # pitch std 
    # stats["energy"][2] # energy mean
    # stats["energy"][3] # energy std
    pitch_path = os.path.join(config["path"]["preprocessed_path"], "pitch/")
    energy_path = os.path.join(config["path"]["preprocessed_path"], "energy/")
    print(pitch_path)
    print(energy_path)
    
    for p in tqdm(os.listdir(pitch_path)):
        p = os.path.join(pitch_path, p)
        try:
            pitch = np.load(p)
            pitch = pitch * stats["pitch"][3] + stats["pitch"][2]
            np.save(p, pitch)
        except Exception as e:
            print("error p: ", p)

    for e in tqdm(os.listdir(energy_path)):
        e = os.path.join(energy_path, e)
        energy = np.load(e)
        energy = energy * stats["energy"][3] + stats["energy"][2]
        np.save(e, energy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)