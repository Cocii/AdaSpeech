import numpy as np
import torch
import torch.nn as nn


class Discriminator2DFactory(nn.Module):
    def __init__(self, time_length, freq_length=80, kernel=(3, 3), c_in=1, hidden_size=128,
                 norm_type='bn', reduction='sum'):
        super(Discriminator2DFactory, self).__init__()
        padding = (kernel[0] // 2, kernel[1] // 2)

        def discriminator_block(in_filters, out_filters, first=False):
            """
            Input: (B, in, 2H, 2W)
            Output:(B, out, H,  W)
            """
            conv = nn.Conv2d(in_filters, out_filters, kernel, (2, 2), padding)
            if norm_type == 'sn':
                conv = nn.utils.spectral_norm(conv)
            block = [
                conv,  # padding = kernel//2
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if norm_type == 'bn' and not first:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            if norm_type == 'in' and not first:
                block.append(nn.InstanceNorm2d(out_filters, affine=True))
            block = nn.Sequential(*block)
            return block

        self.model = nn.ModuleList([
            discriminator_block(c_in, hidden_size, first=True),
            discriminator_block(hidden_size, hidden_size),
            discriminator_block(hidden_size, hidden_size),
        ])

        self.reduction = reduction
        ds_size = (time_length // 2 ** 3, (freq_length + 7) // 2 ** 3)
        if reduction != 'none':
            # The height and width of downsampled image
            self.adv_layer = nn.Linear(hidden_size * ds_size[0] * ds_size[1], 1)
        else:
            self.adv_layer = nn.Linear(hidden_size * ds_size[1], 1)

    def forward(self, x):
        """

        :param x: [B, C, T, n_bins]
        :return: validity: [B, 1], h: List of hiddens
        """
        h = []
        for l in self.model:
            x = l(x)
            h.append(x)
        if self.reduction != 'none':
            x = x.view(x.shape[0], -1)
            validity = self.adv_layer(x)  # [B, 1]
        else:
            B, _, T_, _ = x.shape
            x = x.transpose(1, 2).reshape(B, T_, -1)
            validity = self.adv_layer(x)[:, :, 0]  # [B, T]
        return validity, h


class MultiWindowDiscriminator(nn.Module):
    def __init__(self, time_lengths, cond_size=0, freq_length=80, kernel=(3, 3),
                 c_in=1, hidden_size=128, norm_type='bn', reduction='sum'):
        super(MultiWindowDiscriminator, self).__init__()
        self.win_lengths = time_lengths
        self.reduction = reduction

        self.conv_layers = nn.ModuleList()
        if cond_size > 0:
            self.cond_proj_layers = nn.ModuleList()
            self.mel_proj_layers = nn.ModuleList()
        for time_length in time_lengths:
            conv_layer = [
                Discriminator2DFactory(
                    time_length, freq_length, kernel, c_in=c_in, hidden_size=hidden_size,
                    norm_type=norm_type, reduction=reduction)
            ]
            self.conv_layers += conv_layer
            if cond_size > 0:
                self.cond_proj_layers.append(nn.Linear(cond_size, freq_length))
                self.mel_proj_layers.append(nn.Linear(freq_length, freq_length))

    def forward(self, x, x_len, cond=None, start_frames_wins=None):
        '''
        Args:
            x (tensor): input mel, (B, c_in, T, n_bins).
            x_length (tensor): len of per mel. (B,).

        Returns:
            tensor : (B).
        '''
        validity = []
        if start_frames_wins is None:
            start_frames_wins = [None] * len(self.conv_layers)
        h = []
        for i, start_frames in zip(range(len(self.conv_layers)), start_frames_wins):
            x_clip, c_clip, start_frames = self.clip(
                x, cond, x_len, self.win_lengths[i], start_frames)  # (B, win_length, C)
            start_frames_wins[i] = start_frames
            if x_clip is None:
                continue
            if cond is not None:
                x_clip = self.mel_proj_layers[i](x_clip)  # (B, 1, win_length, C)
                c_clip = self.cond_proj_layers[i](c_clip)[:, None]  # (B, 1, win_length, C)
                x_clip = x_clip + c_clip
            x_clip, h_ = self.conv_layers[i](x_clip)
            h += h_
            validity.append(x_clip)
        if len(validity) != len(self.conv_layers):
            return None, start_frames_wins, h
        if self.reduction == 'sum':
            validity = sum(validity)  # [B]
        elif self.reduction == 'stack':
            validity = torch.stack(validity, -1)  # [B, W_L]
        elif self.reduction == 'none':
            validity = torch.cat(validity, -1)  # [B, W_sum]
        return validity, start_frames_wins, h

    def clip(self, x, cond, x_len, win_length, start_frames=None):
        '''Ramdom clip x to win_length.
        Args:
            x (tensor) : (B, c_in, T, n_bins).
            cond (tensor) : (B, T, H).
            x_len (tensor) : (B,).
            win_length (int): target clip length

        Returns:
            (tensor) : (B, c_in, win_length, n_bins).

        '''
        T_start = 0
        T_end = x_len.max() - win_length
        if T_end < 0:
            return None, None, start_frames
        T_end = T_end.item()
        if start_frames is None:
            start_frame = np.random.randint(low=T_start, high=T_end + 1)
            start_frames = [start_frame] * x.size(0)
        else:
            start_frame = start_frames[0]
        x_batch = x[:, :, start_frame: start_frame + win_length]
        c_batch = cond[:, start_frame: start_frame + win_length] if cond is not None else None
        return x_batch, c_batch, start_frames


class Discriminator(nn.Module):
    def __init__(self, time_lengths=[32, 64, 128], freq_length=80, cond_size=0, kernel=(3, 3), c_in=1,
                 hidden_size=128, norm_type='bn', reduction='sum', uncond_disc=True):
        super(Discriminator, self).__init__()
        self.time_lengths = time_lengths
        self.cond_size = cond_size
        self.reduction = reduction
        self.uncond_disc = uncond_disc
        if uncond_disc:
            self.discriminator = MultiWindowDiscriminator(
                freq_length=freq_length,
                time_lengths=time_lengths,
                kernel=kernel,
                c_in=c_in, hidden_size=hidden_size, norm_type=norm_type,
                reduction=reduction
            )
        if cond_size > 0:
            self.cond_disc = MultiWindowDiscriminator(
                freq_length=freq_length,
                time_lengths=time_lengths,
                cond_size=cond_size,
                kernel=kernel,
                c_in=c_in, hidden_size=hidden_size, norm_type=norm_type,
                reduction=reduction
            )

    def forward(self, x, cond=None, start_frames_wins=None):
        """

        :param x: [B, T, 80]
        :param cond: [B, T, cond_size]
        :param return_y_only:
        :return:
        """
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        x_len = x.sum([1, -1]).ne(0).int().sum([-1])
        ret = {'y_c': None, 'y': None}
        if self.uncond_disc:
            ret['y'], start_frames_wins, ret['h'] = self.discriminator(
                x, x_len, start_frames_wins=start_frames_wins)
        if self.cond_size > 0 and cond is not None:
            ret['y_c'], start_frames_wins, ret['h_c'] = self.cond_disc(
                x, x_len, cond, start_frames_wins=start_frames_wins)
        ret['start_frames_wins'] = start_frames_wins
        return ret['y']

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


if __name__ == "__main__":
    import argparse
    import yaml
    import sys
    sys.path.append("/workspace/nartts/AdaSpeech")
    from utils.model import get_model, get_param_num, get_disc
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p", "--preprocess_config",
        default="/workspace/nartts/AdaSpeech/config/en_cn_spkr/preprocess.yaml",
        type=str, help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", 
        default="/workspace/nartts/AdaSpeech/config/en_cn_spkr/model.yaml", 
        type=str, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", 
        default="/workspace/nartts/AdaSpeech/config/en_cn_spkr/train.yaml", 
        type=str, help="path to train.yaml"
    )

    args = parser.parse_args()
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    d_model, d_optimizer = get_disc(args, configs, torch.device('cuda:0'), train=True)


    a = torch.randn(2, 150, 80).to(torch.device('cuda:0'))
    o = d_model(a)
    for i in range(1, 2000):
        a = torch.randn(2, i, 80).to(torch.device('cuda:0'))
        o = d_model(a)
        if o == None:
            print("i: ", i, "o: ", o)

    # for i in range(200):
    #     d_optimizer.zero_grad()
    #     d_optimizer.step_and_update_lr()
    #     for param_group in d_optimizer._optimizer.param_groups:
    #         print(i, param_group['lr'])

    print("Test done.")