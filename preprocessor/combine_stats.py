import json
import os
import numpy as np
from tqdm import tqdm
import argparse
import yaml
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

stats_path = [
    "/workspace/nartts/AdaSpeech/preprocessed_data_libri_spkr/stats.json",
    "/data/yangyitao/preprocessed_data/cctv_212018_yw/stats.json"
]
stats = []
for i in stats_path:
    with open(os.path.join(i), "r") as f:
        stats.append(json.load(f))


with open(os.path.join(stats_path[0], "stats_combination.json"), "w") as f:
    json.dump(stats, f, cls=MyEncoder)