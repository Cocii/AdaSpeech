from distutils.command.config import config
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.util import normalize
import soundfile as sf
from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from .adaspeech_modules import UtteranceEncoder, PhonemeLevelEncoder, PhonemeLevelPredictor, Condional_LayerNorm
from utils.tools import get_mask_from_lengths
from tqdm import tqdm
# from pyannote.audio import Inference
# from pyannote.audio import Model

# speakers don't have a unique id
class AdaSpeech_spkr_en(nn.Module):
    """ AdaSpeech """
    def __init__(self, preprocess_config, model_config):
        super(AdaSpeech_spkr_en, self).__init__()
        self.model_config = model_config
        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.UtteranceEncoder = UtteranceEncoder(model_config)
        self.PhonemeLevelEncoder = PhonemeLevelEncoder(model_config)
        self.PhonemeLevelPredictor = PhonemeLevelPredictor(model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        # model = Model.from_pretrained("pyannote/embedding",use_auth_token="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")
        # self.speaker_emb = Inference(model, window="whole")
        self.spk_norm = nn.LayerNorm(512)
        self.spk_linear = nn.Linear(512, model_config["transformer"]["encoder_hidden"])
        
        # self.speaker_emb = nn.Embedding(
        #     model_config["language_speaker"]["num_speaker"],
        #     model_config["transformer"]["encoder_hidden"]
        # )
        # with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
        #     self.speaker_map = json.load(f)
        
        self.phone_level_embed = nn.Linear(
            model_config["PhoneEmbedding"]["phn_latent_dim"],
            model_config["PhoneEmbedding"]["adim"]
        )
        self.lang_emb = nn.Embedding(
            model_config["language_speaker"]["num_language"],
            model_config["transformer"]["encoder_hidden"]
        )
        self.layer_norm = Condional_LayerNorm(preprocess_config["preprocessing"]["mel"]["n_mel_channels"])
        self.postnet = PostNet()
        self.transform_layer_dim = model_config["transformer"]["encoder_hidden"]
        self.dataset_name = preprocess_config["dataset"]
        self.datasets_base_path = preprocess_config["path"]["datasets_base_path"]
        self.spkr_emb_path = []
        for d in self.dataset_name:
            self.spkr_emb_path.append(os.path.join(self.datasets_base_path, d, "speaker_embed/"))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(
        self,
        ids,
        raw_texts,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        avg_targets=None,
        languages=None,
        phoneme_level_predictor=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        assert len(speakers) == len(ids)
        embeddings = []
        for i in range(len(speakers)):
            if len(self.spkr_emb_path) > 1:
                embedding_path = os.path.join(self.spkr_emb_path[int(languages[i])], f"{speakers[i]}-spk-{ids[i]}.npy")
            else:
                embedding_path = os.path.join(self.spkr_emb_path[0], f"{speakers[i]}-spk-{ids[i]}.npy")
            if not os.path.exists(embedding_path):
                print("adaspeech.py === embedding_path: ",embedding_path)
            assert os.path.exists(embedding_path)
            emb_np = np.load(embedding_path)
            emb_tensor = torch.from_numpy(emb_np).float()
            emb_tensor = emb_tensor.to(self.device)
            embeddings.append(emb_tensor)
        # Stack embeddings
        speaker_embedding = torch.stack(embeddings)
        speaker_embedding = self.spk_norm(speaker_embedding)
        speaker_embedding = self.spk_linear(speaker_embedding)
        language_embedding = self.lang_emb(languages)

        output = self.encoder(texts, speaker_embedding, src_masks)
        xs = self.UtteranceEncoder(torch.transpose(mels, 1, 2))
        xs = torch.transpose(xs, 1, 2)
        output = output + xs.expand(-1, max_src_len, -1)

        if phoneme_level_predictor:
            phn_predict = self.PhonemeLevelPredictor(output.transpose(1, 2))
            with torch.no_grad():
                phn_encode = self.PhonemeLevelEncoder(avg_targets.transpose(1, 2))
            if output.shape != self.phone_level_embed(phn_encode).shape:
                print("phn_encode: ", phn_encode)
                print("output.shape: ", output.shape)
                print("self.phone_level_embed(phn_encode).shape: ", self.phone_level_embed(phn_encode).shape)
            assert output.shape == self.phone_level_embed(phn_encode).shape
            output = output + self.phone_level_embed(phn_encode.detach())
        else:
            phn_predict = self.PhonemeLevelPredictor(output.transpose(1, 2))
            phn_encode = self.PhonemeLevelEncoder(avg_targets.transpose(1, 2))
            if output.shape != self.phone_level_embed(phn_encode).shape:
                print("phn_encode: ", phn_encode)
                print("output.shape: ", output.shape)
                print("self.phone_level_embed(phn_encode).shape: ", self.phone_level_embed(phn_encode).shape)
            assert output.shape == self.phone_level_embed(phn_encode).shape
            output = output + self.phone_level_embed(phn_encode)

        output = output + speaker_embedding.unsqueeze(1).expand(
            -1, max_src_len, -1
        )

        output = output + language_embedding.unsqueeze(1).expand(
            -1, max_src_len, -1
        )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, speaker_embedding, mel_masks)
        output = self.mel_linear(output)
        output = self.layer_norm(output, speaker_embedding)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            phn_predict,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            phn_encode,
        )

    def inference(
        self,
        ids,
        raw_texts,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        languages=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        avg_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        embeddings = []
        speakers_dic = {}
        if len(self.dataset_name) > 1:
            with open(os.path.join(self.datasets_base_path, self.dataset_name[0], "speakers_combine.json"), "r") as f:
                speakers_dic = json.load(f)
        else:
            with open(os.path.join(self.datasets_base_path, self.dataset_name[0], "speakers.json"), "r") as f:
                speakers_dic = json.load(f)

        for i in range(len(speakers)):
            # print("speakers[i]", speakers[i])
            # print("speakers_dic", speakers_dic)
            speaker_str = next((key for key, value in speakers_dic.items() if value == speakers[i]), None)
            try:
                speaker = speaker_str.split('_')[0]
            except:
                print("Check if there is dataset name following speaker_id in inference.sh!")
                
            
            if len(self.spkr_emb_path)>1:
                embedding_path = os.path.join(self.spkr_emb_path[int(languages[i])], f"{speaker}-spk-{ids[i]}.npy")
            else:
                embedding_path = os.path.join(self.spkr_emb_path[0], f"{speaker}-spk-{ids[i]}.npy")
            
            if not os.path.exists(embedding_path):
                print("adaspeech.py === embedding_path: ",embedding_path)
            assert os.path.exists(embedding_path)

            if len(ids[i]) >= 21:
                emb_np = np.load(embedding_path)
                emb_tensor = torch.from_numpy(emb_np).float()
            else:
                emb_tensor = torch.load(embedding_path)
            emb_tensor = emb_tensor.to(self.device)
            embeddings.append(emb_tensor)
        # Stack embeddings
        speaker_embedding = torch.stack(embeddings)
        speaker_embedding = self.spk_norm(speaker_embedding)
        speaker_embedding = self.spk_linear(speaker_embedding)

        
        output = self.encoder(texts, speaker_embedding, src_masks)

        # mel of reference audios
        # xs = self.UtteranceEncoder(torch.transpose(mels, 1, 2))
        # xs = torch.transpose(xs, 1, 2)
        # output = output + xs.expand(-1, max_src_len, -1)

        phn_predict = self.PhonemeLevelPredictor(output.transpose(1, 2))
        phn_encode = None
        output = output + self.phone_level_embed(phn_predict)

        output = output + speaker_embedding.unsqueeze(1).expand(
            -1, max_src_len, -1
        )

        # language embedding part
        language_embedding = self.lang_emb(languages)
        output = output + language_embedding.unsqueeze(1).expand(-1, max_src_len, -1)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, speaker_embedding, mel_masks)
        output = self.mel_linear(output)
        output = self.layer_norm(output, speaker_embedding)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            phn_predict,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            phn_encode,
        )
        
