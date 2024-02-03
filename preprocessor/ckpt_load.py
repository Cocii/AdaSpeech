import torch
import yaml
import sys
sys.path.append("/workspace/nartts/AdaSpeech/")
from model import AdaSpeech_spkr_en_cn, ScheduledOptim
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_optimizer(scheduled_optim):
    total_size = 0
    for param_group in scheduled_optim._optimizer.param_groups:
        for param in param_group['params']:
            total_size += param.numel()
    return total_size



# "/workspace/nartts/AdaSpeech/output/output_en_cn/ckpt/adaspeech_en_cn/162000.pth.tar"
# "/workspace/nartts/AdaSpeech/output/output_finetune_trump/ckpt/finetune_trump/1000.pth.tar"
preprocess_config = yaml.load(open("/workspace/nartts/AdaSpeech/config/finetune_trump/preprocess.yaml", "r"), Loader=yaml.FullLoader)
model_config = yaml.load(open("/workspace/nartts/AdaSpeech/config/finetune_trump/model.yaml", "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open("/workspace/nartts/AdaSpeech/config/finetune_trump/train.yaml", "r"), Loader=yaml.FullLoader)

device = torch.device("cpu")


ckpt_path = "/workspace/nartts/AdaSpeech/output/output_en_cn/ckpt/adaspeech_en_cn/162000.pth.tar"
fine_path = "/workspace/nartts/AdaSpeech/output/output_finetune_trump/ckpt/finetune_trump/1000.pth.tar"
ckpt = torch.load(ckpt_path)
fine = torch.load(fine_path)
model = AdaSpeech_spkr_en_cn(preprocess_config, model_config).to(device)

model.load_state_dict(ckpt["model"])
finet = AdaSpeech_spkr_en_cn(preprocess_config, model_config).to(device)
finet.load_state_dict(fine["model"])


optimizer0 = ScheduledOptim(
        model, train_config, model_config, 0
    )
optimizer1 = ScheduledOptim(
        finet, train_config, model_config, 0
    )
optimizer0.load_state_dict(ckpt["optimizer"])
optimizer1.load_state_dict(fine["optimizer"])
a = [a[0] for a in model.named_parameters()]
b = [b[0] for b in finet.named_parameters()]


model_size = count_parameters(model)
finet_size = count_parameters(finet)

model_size = count_optimizer(optimizer0)
finet_size = count_optimizer(optimizer1)
print(f"Model size: {model_size}")
print(f"Model size (finetune): {finet_size}")

print(model.named_parameters())
print(finet.named_parameters())