import torch
import os
import sys
sys.path.append("/workspace/nartts/AdaSpeech/")
from model import AdaSpeech_spkr_en_cn
import yaml

ckpt = torch.load('/workspace/nartts/AdaSpeech/output/output_en_cn/ckpt/adaspeech_en_cn/162000.pth.tar')
model_config = yaml.load(open("/workspace/nartts/AdaSpeech/config/finetune_xjp/model.yaml", "r"), Loader=yaml.FullLoader)
preprocess_config = yaml.load(open("/workspace/nartts/AdaSpeech/config/finetune_xjp/preprocess.yaml", "r"), Loader=yaml.FullLoader)
model = AdaSpeech_spkr_en_cn(preprocess_config, model_config)
model.load_state_dict(ckpt["model"])

# print(model)

for name, param in model.named_parameters():
    print(name, param.requires_grad)
