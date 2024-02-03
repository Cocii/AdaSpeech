import torch
from torch import nn
from xlsr.model.wav2vec2_encoder import load_model, extract_xlsr_spkr


class Conv1D_ReLU_BN(nn.Module):
    def __init__(self, c_in, c_out, ks, stride, padding, dilation):
        super(Conv1D_ReLU_BN, self).__init__()

        self.network = nn.Sequential(
            nn.Conv1d(c_in, c_out, ks, stride, padding, dilation),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(c_out),
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Res2_Conv1D(nn.Module):
    def __init__(self, c, scale, ks, stride, padding, dilation):
        super(Res2_Conv1D, self).__init__()
        assert c % scale == 0
        self.c = c
        self.scale = scale
        self.width = c // scale

        self.convs = []
        self.bns = []

        for i in range(scale - 1):
            self.convs.append(nn.Conv1d(self.width, self.width, ks, stride, padding, dilation))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        """
        param x: (B x c x d)
        """

        xs = torch.split(x, self.width, dim=1)  # channel-wise split
        ys = []

        for i in range(self.scale):
            if i == 0:
                x_ = xs[i]
                y_ = x_
            elif i == 1:
                x_ = xs[i]
                y_ = self.bns[i - 1](self.convs[i - 1](x_))
            else:
                x_ = xs[i] + ys[i - 1]
                y_ = self.bns[i - 1](self.convs[i - 1](x_))
            ys.append(y_)

        y = torch.cat(ys, dim=1)  # channel-wise concat
        return y



class Res2_Conv1D_ReLU_BN(nn.Module):
    def __init__(self, channel, scale, ks, stride, padding, dilation):
        super(Res2_Conv1D_ReLU_BN, self).__init__()

        self.network = nn.Sequential(
            Res2_Conv1D(channel, scale, ks, stride, padding, dilation),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(channel),
        )

    def forward(self, x):
        y = self.network(x)
        return y


class SE_Block(nn.Module):
    def __init__(self, c_in, c_mid):
        super(SE_Block, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(c_in, c_mid),
            nn.ReLU(inplace=True),
            nn.Linear(c_mid, c_in),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.network(x.mean(dim=-1))
        y = x * s.unsqueeze(-1)
        return y


class SE_Res2_Block(nn.Module):
    def __init__(self, channel, scale, ks, stride, padding, dilation):
        super(SE_Res2_Block, self).__init__()
        self.network = nn.Sequential(
            Conv1D_ReLU_BN(channel, channel, 1, 1, 0, 1),
            Res2_Conv1D_ReLU_BN(channel, scale, ks, stride, padding, dilation),
            Conv1D_ReLU_BN(channel, channel, 1, 1, 0, 1),
            SE_Block(channel, channel)
        )

    def forward(self, x):
        y = self.network(x) + x
        return y


class AttentiveStatisticPool(nn.Module):
    def __init__(self, c_in, c_mid):
        super(AttentiveStatisticPool, self).__init__()

        self.network = nn.Sequential(
            nn.Conv1d(c_in, c_mid, kernel_size=1),
            nn.Tanh(),  # seems like most implementations uses tanh?
            nn.Conv1d(c_mid, c_in, kernel_size=1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x.shape: B x C x t
        alpha = self.network(x)
        mu_hat = torch.sum(alpha * x, dim=-1)
        var = torch.sum(alpha * x ** 2, dim=-1) - mu_hat ** 2
        std_hat = torch.sqrt(var.clamp(min=1e-9))
        y = torch.cat([mu_hat, std_hat], dim=-1)
        # y.shape: B x (c_in*2)
        return y


class ECAPA_TDNN(nn.Module):
    def __init__(self, c_in=80, c_mid=512, c_out=192):
        super(ECAPA_TDNN, self).__init__()

        self.layer1 = Conv1D_ReLU_BN(c_in, c_mid, 5, 1, 2, 1)
        self.layer2 = SE_Res2_Block(c_mid, 8, 3, 1, 2, 2)
        self.layer3 = SE_Res2_Block(c_mid, 8, 3, 1, 3, 3)
        self.layer4 = SE_Res2_Block(c_mid, 8, 3, 1, 4, 4)

        self.network = nn.Sequential(
            # Figure 2 in https://arxiv.org/pdf/2005.07143.pdf seems like groupconv?
            nn.Conv1d(c_mid * 3, 1536, kernel_size=1, groups=3),
            AttentiveStatisticPool(1536, 128),
        )

        self.ln1 = nn.LayerNorm(3072)
        self.bn1 = nn.BatchNorm1d(3072)
        self.linear = nn.Linear(3072, c_out)
        self.ln2 = nn.LayerNorm(c_out)
        self.bn2 = nn.BatchNorm1d(c_out)

    def forward(self, x):
        # x.shape: B x C x t
        y1 = self.layer1(x)
        y2 = self.layer2(y1) + y1
        y3 = self.layer3(y1 + y2) + y1 + y2
        y4 = self.layer4(y1 + y2 + y3) + y1 + y2 + y3

        y = torch.cat([y2, y3, y4], dim=1)  # channel-wise concat
        y = self.network(y)
        if y.shape[0] == 1:
            y = self.ln1(y.squeeze(-1))
            y = self.linear(y) 
            y = self.ln2(y.squeeze(-1))
        else:
            y = self.linear(self.bn1(y.unsqueeze(-1)).squeeze(-1)) 
            y = self.bn2(y.unsqueeze(-1)).squeeze(-1)

        return y


class SpeakerEncoder(torch.nn.Module):
    def __init__(self, conf=None):
        """We train a speaker embedding network that uses the 1st layer of XLSR-53 as an input. For the speaker embedding network, we borrow the neural architecture from a state-of-the-art speaker recognition network [14]

        Args:
            conf:
        """
        super(SpeakerEncoder, self).__init__()
        self.conf = conf

        self.wav2vec2 = load_model()
        self.spk = ECAPA_TDNN(c_in=1024, c_mid=512, c_out=256)

    def forward(self, x):
        """

        Args:
            x: torch.Tensor of shape (B x t)

        Returns:
            y: torch.Tensor of shape (B x 192)
        """
        y = extract_xlsr_spkr(self.wav2vec2, x)
        y = y.permute((0, 2, 1))  # B x t x C -> B x C x t
        y = self.spk(y)  # B x C(1024) x t -> B x D(192)
        y = torch.nn.functional.normalize(y, p=2, dim=-1)
        return y
            
import os
import torchaudio
from tqdm import tqdm
import numpy as np
import random
import librosa

def get_wavs(root_dir):
    subdirectories = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                subdirectories.append(os.path.join(root, file))
    return subdirectories

def cctv_get_wavs(root_dir):
    dirs = ['2018', '2020', '2021']
    subdirectories = []
    for d in dirs:
        for root, mid, files in os.walk(os.path.join(root_dir, d)):
            for file in files:
                if file.endswith(".wav"):
                    subdirectories.append(os.path.join(root, file))
    return subdirectories

def libri_split(w):
    speaker, id = w.split('/')[-1].split('.')[0].split('_', 1)
    return speaker, id

def aishell_split(w):
    speaker = w.split('/')[-2]
    id = w.split('/')[-1].split('.')[0]
    return speaker, id

def cctv_split(w):
    speaker = w.split('/')[-5]
    id = w.split('/')[-1].split('.')[0]
    return speaker, id

def multi_infer(all_wavs, output_path):
    if output_path.split('/')[-2] == 'libri_spkr':
        data_split = libri_split
    if output_path.split('/')[-2] == 'aishell3':
        data_split = aishell_split
    if output_path.split('/')[-2] == 'cctv_212018_yw':
        data_split = cctv_split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # its = reversed(range(len(all_wavs)))
    random.shuffle(all_wavs)
    its = range(len(all_wavs))
    spk_encoder = SpeakerEncoder().to(device)
    for i in tqdm(its):
        i = len(all_wavs)-1-i
        w = all_wavs[i]
        speaker, id = data_split(w)
        save_path = os.path.join(output_path, speaker+'-xlsr-'+id+'.npy')
        if not os.path.exists(save_path):
            data, _ = librosa.load(w, sr=16000)
            data = np.expand_dims(data, axis=0)
            x = torch.from_numpy(data)
            if x.shape[1] / 16000 > 30:
                continue
            else:
                x = x.to(device)
                a = spk_encoder(x).detach().cpu().numpy()
                np.save(save_path, a)

def cctv_infer():
    with open("/data/speech_data/cuijiayan/tools/xlsr/text.txt", 'r') as f:
        lines = f.readlines()
    all_wavs = [l.strip('\n') for l in lines]
    output_path = "/workspace/nartts/AdaSpeech/preprocessed_data/cctv_212018_yw/xlsr"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spk_encoder = SpeakerEncoder().to(device)
    # all_wavs = reversed(all_wavs)
    for wav_path in tqdm(all_wavs):
        speaker = wav_path.split('/')[-5]
        id = wav_path.split('/')[-1].split('.')[0]
        save_path = os.path.join(output_path, speaker+'-xlsr-'+id+'.npy')
        if not os.path.exists(save_path):
            data, _ = librosa.load(wav_path, sr=16000)
            data = np.expand_dims(data, axis=0)
            x = torch.from_numpy(data)
            x = x.to(device)
            a = spk_encoder(x).detach().cpu().numpy()
            np.save(save_path, a)

def libri_infer():
    with open("/data/speech_data/cuijiayan/tools/xlsr/text.txt", 'r') as f:
        lines = f.readlines()
    all_wavs = [l.strip('\n') for l in lines]
    output_path = "/workspace/nartts/AdaSpeech/preprocessed_data/libri_spkr/xlsr"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spk_encoder = SpeakerEncoder().to(device)
    # all_wavs = reversed(all_wavs)
    for wav_path in tqdm(all_wavs):
        speaker = wav_path.split('/')[-1].split('_')[0]
        id = wav_path.split('/')[-1].split('_', 1)[-1].split('.')[0]
        save_path = os.path.join(output_path, speaker+'-xlsr-'+id+'.npy')
        if not os.path.exists(save_path):
            data, _ = librosa.load(wav_path, sr=16000)
            data = np.expand_dims(data, axis=0)
            x = torch.from_numpy(data)
            x = x.to(device)
            a = spk_encoder(x).detach().cpu().numpy()
            np.save(save_path, a)
 
if __name__ == '__main__':
    # sampling rate should be 16000
    all_wavs = get_wavs("/data/speech_data/LibriTTS/audios/en")
    # all_wavs = get_wavs("/data/yangyitao/preprocessed_data/aishell3/wav")
    # all_wavs = cctv_get_wavs("/data/speech_data/cctv_cjy/")
    
    output_path = "/workspace/nartts/AdaSpeech/preprocessed_data/libri_spkr/xlsr"
    # output_path = "/workspace/nartts/AdaSpeech/preprocessed_data/aishell3/xlsr"
    # output_path = "/workspace/nartts/AdaSpeech/preprocessed_data/cctv_212018_yw/xlsr"
    
    # multi_infer(all_wavs, output_path)
    # cctv_infer()
    libri_infer()