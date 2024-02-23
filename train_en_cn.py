import argparse
from email.generator import Generator
import json
import os
from sklearn.utils import shuffle

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_param_num, get_disc
from utils.tools import to_device, log, synth_one_sample, AttrDict, infer_mels
from model import AdaSpeechLoss
from dataset import Dataset

from evaluate import evaluate
import sys
sys.path.append("vocoder")
import numpy as np
import soundfile as sf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(1234)



def get_vocoder(config, checkpoint_path):
    if checkpoint_path.split("/")[-2] == "hifigan":
        from vocoder.models.hifigan import Generator
    elif checkpoint_path.split("/")[-2] == "bigvgan_22khz_80band" or checkpoint_path.split("/")[-2] == "BigVGAN":
        from vocoder.models.BigVGAN import BigVGAN as Generator

    config = json.load(open(config, 'r', encoding='utf-8'))
    config = AttrDict(config)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    vocoder = Generator(config).to(device).eval()
    vocoder.load_state_dict(checkpoint_dict['generator'])
    vocoder.remove_weight_norm()
    return vocoder

def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    # dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    dataset = Dataset(
        "train_remove_short.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset

    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        shuffle = True,
        batch_size=batch_size * group_size,
        collate_fn=dataset.collate_fn,
        num_workers=4,
    )

    # Prepare model
    #
    model, optimizer = get_model(args, configs, device, train=True)
    d_model, d_optimizer = get_disc(args, configs, device, train=True)
    MSELoss = nn.MSELoss()
    model = nn.DataParallel(model)
    d_model = nn.DataParallel(d_model)
    num_param = get_param_num(model)
    Loss = AdaSpeechLoss(preprocess_config, model_config).to(device)
    print("Number of AdaSpeech Parameters:", num_param)

    # Load vocoder
    #vocoder = get_vocoder(model_config, device)
    vocoder = get_vocoder(args.vocoder_config, args.vocoder_checkpoint)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    os.makedirs(os.path.join(train_config["path"]["ckpt_path"], 'disc'), exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    phoneme_level_encoder_step = train_config["step"]["phoneme_level_encoder_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()
    
    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            # batch = 
            #   0           1        2      3          4              5      6         7             8         9         10       11           12         13           14
            # ids, raw_texts, speakers, texts, text_lens, max(text_lens), mels, mel_lens, max(mel_lens), pitches, energies, durations, avg_mel_phs, spk_embs, language_ids
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward
                if step >= phoneme_level_encoder_step:
                    phoneme_level_predictor = True
                    exe_batch = batch + (phoneme_level_predictor, )
                    output = model(*(exe_batch))
                else:
                    phoneme_level_predictor = False
                    exe_batch = batch + (phoneme_level_predictor, )
                    output = model(*(exe_batch))

                if train_config["infer"] and epoch == 1:
                    mels, wav_reconstructions, tags = infer_mels(
                            batch,
                            output,
                            vocoder,
                            model_config,
                            preprocess_config
                        )
                    for i in range(len(mels)):
                        mel_path = os.path.join(train_config["path"]["save_path"], "mels", tags[i]+".npy")
                        wav_path = os.path.join(train_config["path"]["save_path"], "audios", tags[i]+".wav")
                        os.makedirs(os.path.dirname(mel_path), exist_ok=True)
                        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
                        np.save(mel_path, mels[i].cpu().numpy())
                        sf.write(wav_path, wav_reconstructions[i], samplerate=22050)
                        
                elif train_config["infer"] and epoch > 1:
                    quit()
                else:
                    mel_predicted = output[1]
                    o_disc = d_model(mel_predicted)
                    p_ = o_disc
                    adv_loss = 0.05 * MSELoss(p_, p_.detach().new_ones(p_.size()))
                    # adv_loss = torch.FloatTensor([0]).to(output[1].device)
                    ###########################
                    #      Discriminator      #
                    ###########################             
                    d_optimizer.zero_grad()
                    mel_gt = batch[6]
                    o_disc_gt = d_model(mel_gt)
                    o_disc_p = d_model(mel_predicted.detach())
                    p_gt = o_disc_gt
                    p_p = o_disc_p
                    d_loss_r = ((p_gt - 1) ** 2).mean()
                    d_loss_f = (p_p ** 2).mean()
                    total_d_loss_rf = d_loss_r + d_loss_f
                    # total_d_loss = torch.FloatTensor([0]).to(output[1].device)
                    # Disc Backward
                    total_d_loss = total_d_loss_rf / grad_acc_step
                    total_d_loss.backward()

                    if step % grad_acc_step == 0:
                        nn.utils.clip_grad_norm_(d_model.parameters(), grad_clip_thresh)
                        d_optimizer.step_and_update_lr()
                        d_optimizer.zero_grad()

                    if step >= phoneme_level_encoder_step:
                        losses = Loss(batch, output, phoneme_level_loss = True)
                    else:
                        losses = Loss(batch, output, phoneme_level_loss = False)
                    total_loss = losses[0]

                    # Generator Backward
                    total_loss = (total_loss + adv_loss)/ grad_acc_step
                    total_loss.backward()

                    if step % grad_acc_step == 0:
                        # Clipping gradients to avoid gradient explosion
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                        # Update weights
                        optimizer.step_and_update_lr()
                        optimizer.zero_grad()

                    if step % log_step == 0:
                        losses = [l.item() for l in losses]
                        message1 = "Step {}/{}, ".format(step, total_step)
                        message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, Phone_Level Loss: {:.4f}".format(
                            *losses
                        )
                        message3 = ", Discriminator Loss: {:.4f}, Adversarial Loss: {:.4f}".format(
                            total_d_loss_rf.item(), adv_loss.item()
                        )
                        with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                            f.write(message1 + message2 + message3 + "\n")

                        outer_bar.write(message1 + message2 + message3)

                        log(train_logger, step, losses=losses+[total_d_loss_rf.item(), adv_loss.item()])

                    if step % synth_step == 0:
                        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                            batch,
                            output,
                            vocoder,
                            model_config,
                            preprocess_config,
                        )
                        log(
                            train_logger,
                            fig=fig,
                            tag="Training/step_{}_{}".format(step, tag),
                        )
                        sampling_rate = preprocess_config["preprocessing"]["audio"][
                            "sampling_rate"
                        ]
                        log(
                            train_logger,
                            audio=wav_reconstruction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_reconstructed".format(step, tag),
                        )
                        log(
                            train_logger,
                            audio=wav_prediction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_synthesized".format(step, tag),
                        )

                    if step % val_step == 0:
                        d_model.eval()
                        model.eval()

                        message = evaluate(d_model, model, step, configs, val_logger, vocoder)
                        with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                            f.write(message + "\n")
                        outer_bar.write(message)

                        d_model.train()
                        model.train()

                    if step % save_step == 0:
                        torch.save(
                            {
                                "model": model.module.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict(),
                            },
                            os.path.join(
                                train_config["path"]["ckpt_path"],
                                "{}.pth.tar".format(step),
                            ),
                        )
                        torch.save(
                            {
                                "model": d_model.module.state_dict(),
                                "optimizer": d_optimizer._optimizer.state_dict(),
                            },
                            os.path.join(
                                train_config["path"]["ckpt_path"],"disc",
                                "{}.pth.tar".format(step),
                            ),
                        )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--vocoder_checkpoint", type=str, default=None, required= True, help="path to vocoder checkpoint"
    )
    parser.add_argument(
        "--vocoder_config", type=str, default=None, required=True, help="path to vocoder config"
    )

    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)


    main(args, configs)
