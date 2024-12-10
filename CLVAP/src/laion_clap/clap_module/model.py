""" CLAP Model

Adapted from CLIP: https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
Adapted to the Audio Task.
"""

from collections import OrderedDict
from dataclasses import dataclass
from email.mime import audio
from typing import Tuple, Union, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .timm_model import TimmModel
import logging
from .utils import freeze_batch_norm_2d

from .pann_model import create_pann_model
from .htsat import create_htsat_model
from transformers import BertModel, RobertaModel, BartModel
from transformers.tokenization_utils_base import BatchEncoding


class MLPLayers(nn.Module):
    def __init__(self, units=[512, 512, 512], nonlin=nn.ReLU(), dropout=0.1):
        super(MLPLayers, self).__init__()
        self.nonlin = nonlin
        self.dropout = dropout

        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]

        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        X = self.sequential(X)
        return X


# Audio Config Class
@dataclass
class CLAPAudioCfp:
    model_type: str = "PANN"
    model_name: str = "Cnn14"
    sample_rate: int = 48000
    # Param
    audio_length: int = 1024
    window_size: int = 1024
    hop_size: int = 1024
    fmin: int = 50
    fmax: int = 14000
    class_num: int = 527
    mel_bins: int = 64
    clip_samples: int = 480000


@dataclass
class CLAPTextCfg:
    context_length: int
    vocab_size: int
    width: int
    heads: int
    layers: int
    model_type: str

@dataclass
class CLAPVocalCfg:
    model_type: str = "HTSAT"
    model_name: str = "Cnn14"
    sample_rate: int = 48000
    audio_length: int = 1024
    window_size: int = 1024
    hop_size: int = 1024
    fmin: int = 50
    fmax: int = 14000
    class_num: int = 527
    mel_bins: int = 64
    clip_samples: int = 480000


class CLVAP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        audio_cfg: CLAPAudioCfp,
        text_cfg: CLAPTextCfg,
        vocal_cfg: CLAPVocalCfg,
        quick_gelu: bool = False,
        enable_fusion: bool = False,
        fusion_type: str = 'None',
        joint_embed_shape: int = 512,
        mlp_act: str = 'relu',
    ):
        super().__init__()
        if isinstance(audio_cfg, dict):
            audio_cfg = CLAPAudioCfp(**audio_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLAPTextCfg(**text_cfg)
        if isinstance(vocal_cfg, dict):
            vocal_cfg = CLAPVocalCfg(**vocal_cfg)

        self.audio_cfg = audio_cfg
        self.text_cfg = text_cfg
        self.vocal_cfg = vocal_cfg
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type
        self.joint_embed_shape = joint_embed_shape
        self.mlp_act = mlp_act

        self.context_length = text_cfg.context_length

        if mlp_act == 'relu':
            mlp_act_layer = nn.ReLU()
        elif mlp_act == 'gelu':
            mlp_act_layer = nn.GELU()
        else:
            raise NotImplementedError

        # Audio branch
        if audio_cfg.model_type == "HTSAT":
            self.audio_branch = create_htsat_model(audio_cfg, enable_fusion, fusion_type)
            self.audio_projection = nn.Sequential(
                    nn.Linear(embed_dim, self.joint_embed_shape),
                    mlp_act_layer,
                    nn.Linear(self.joint_embed_shape, self.joint_embed_shape)
                )
            self.audio_transform = MLPLayers(units=[self.joint_embed_shape,
                                                    self.joint_embed_shape,
                                                    self.joint_embed_shape], dropout=0.1)
        else:
            logging.error(f"Model config for {audio_cfg.model_type} not found")
            raise RuntimeError(f"Model config for {audio_cfg.model_type} not found.")

        # Text branch
        if text_cfg.model_type == "roberta":
            self.text_branch = RobertaModel.from_pretrained('roberta-base')
            self.text_projection = nn.Sequential(
                nn.Linear(768, self.joint_embed_shape),
                mlp_act_layer,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape)
            )
            # self.text_transform = MLPLayers(units=[self.joint_embed_shape,
            #                                        self.joint_embed_shape,
            #                                        self.joint_embed_shape], dropout=0.1)
        else:
            logging.error(f"Model config for {text_cfg.model_type} not found")
            raise RuntimeError(f"Model config for {text_cfg.model_type} not found.")
        self.text_branch_type = text_cfg.model_type

        # Vocal branch
        if vocal_cfg.model_type == "HTSAT":
            self.vocal_branch = create_htsat_model(vocal_cfg, False, 'None')
            self.vocal_projection = nn.Sequential(
                    nn.Linear(embed_dim, self.joint_embed_shape),
                    mlp_act_layer,
                    nn.Linear(self.joint_embed_shape, self.joint_embed_shape)
                )
            # self.vocal_transform = MLPLayers(units=[self.joint_embed_shape,
            #                                         self.joint_embed_shape,
            #                                         self.joint_embed_shape], dropout=0.1)
        else:
            logging.error(f"Model config for {vocal_cfg.model_type} not found")
            raise RuntimeError(f"Model config for {vocal_cfg.model_type} not found.")

        # Fusion and projection
        self.cross_attention_text_to_vocal = nn.MultiheadAttention(joint_embed_shape, num_heads=8, dropout=0.1)
        self.cross_attention_vocal_to_text = nn.MultiheadAttention(joint_embed_shape, num_heads=8, dropout=0.1)
        self.fusion_mlp = MLPLayers(units=[joint_embed_shape, joint_embed_shape, joint_embed_shape], dropout=0.1)

        # ============================================================================================================

        self.logit_scale_a = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_t = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer("attn_mask", self.build_attention_mask(), persistent=False)

        self.init_text_branch_parameters()

    def init_text_branch_parameters(self):
        if self.text_branch_type == "roberta":
            width = self.text_branch.embeddings.word_embeddings.weight.shape[-1]
        else:
            width = self.text_branch.width
        nn.init.constant_(self.logit_scale_a, np.log(1 / 0.07))
        nn.init.constant_(self.logit_scale_t, np.log(1 / 0.07))

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_audio(self, audio, device):
        return self.audio_branch(audio, mixup_lambda=None, device=device)  # mix lambda needs to add

    def encode_text(self, text, device):
        if self.text_branch_type == "roberta":
            x = self.text_branch(
                input_ids=text["input_ids"].to(device=device, non_blocking=True),
                attention_mask=text["attention_mask"].to(
                    device=device, non_blocking=True
                ),
            )["pooler_output"]
        else:
            logging.error(f"Model type {self.text_branch_type} not found")
            raise RuntimeError(f"Model type {self.text_branch_type} not found.")
        return x
    
    def encode_vocal(self, vocal, device):
        return self.vocal_branch(vocal, mixup_lambda=None, device=device)

    def fuse_text_vocal(self, text_features, vocal_features, device):
        text_features = text_features.unsqueeze(0)
        vocal_features = vocal_features.unsqueeze(0)
        
        attn_output_text_to_vocal, _ = self.cross_attention_text_to_vocal(
            text_features, vocal_features, vocal_features, attn_mask=self.attn_mask.to(device)
        )
        
        attn_output_vocal_to_text, _ = self.cross_attention_vocal_to_text(
            vocal_features, text_features, text_features, attn_mask=self.attn_mask.to(device)
        )
        
        combined_features = (attn_output_text_to_vocal + attn_output_vocal_to_text) / 2
        return combined_features.squeeze(0)


    def forward(self, audio, text, vocal, device=None):
        """Forward audio and text into the CLAP"""
        if device is None:
            if audio is not None:
                device = audio.device
            elif text is not None:
                device = text.device
            elif vocal is not None:
                device = vocal.device
        if audio is None and text is None and vocal is None:
            return self.logit_scale_a.exp(), self.logit_scale_t.exp()
        elif audio is None:
            text_features = self.encode_text(text, device=device)
            text_features = self.text_projection(text_features)
            text_features = F.normalize(text_features, dim=-1)
            vocal_features = self.encode_vocal(vocal, device=device)["embedding"]
            vocal_features = self.vocal_projection(vocal_features)
            vocal_features = F.normalize(vocal_features, dim=-1)
            fused_features = self.fuse_text_vocal(text_features, vocal_features, device)
            return fused_features
        elif text is None:
            audio_features = self.encode_audio(audio, device=device)["embedding"]
            audio_features = self.audio_projection(audio_features)
            return audio_features
        audio_features = self.encode_audio(audio, device=device)["embedding"]
        audio_features = self.audio_projection(audio_features)
        audio_features = F.normalize(audio_features, dim=-1)

        text_features = self.encode_text(text, device=device)
        text_features = self.text_projection(text_features)
        text_features = F.normalize(text_features, dim=-1)
        vocal_features = self.encode_vocal(vocal, device=device)["embedding"]
        vocal_features = self.vocal_projection(vocal_features)
        vocal_features = F.normalize(vocal_features, dim=-1)
        fused_features = self.fuse_text_vocal(text_features, vocal_features, device)

        audio_features_mlp = self.audio_transform(audio_features)
        fused_features_mlp = self.fusion_mlp(fused_features)
        # Four outputs: audio features (basic & MLP), text features (basic & MLP)
        return (
            audio_features,
            text_features,
            audio_features_mlp,
            fused_features_mlp,
            self.logit_scale_a.exp(),
            self.logit_scale_t.exp(),
        )

    def get_logit_scale(self):
        return self.logit_scale_a.exp(), self.logit_scale_t.exp()

    def get_text_embedding(self, data):
        """Get the text embedding from the model

        Parameters
        ----------
        data: torch.Tensor 
            a tensor of text embedding

        Returns
        ----------
        text_embed: torch.Tensor
            a tensor of text_embeds (N, D)

        """
        device = next(self.parameters()).device
        for k in data:
            data[k] = data[k].to(device)
        text_embeds = self.encode_text(data, device=device)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        return text_embeds

    def get_audio_embedding(self, data):
        """Get the audio embedding from the model

        Parameters
        ----------
        data: a list of dict
            the audio input dict list from 'get_audio_feature' method

        Returns
        ----------
        audio_embed: torch.Tensor
            a tensor of audio_embeds (N, D)

        """
        device = next(self.parameters()).device
        input_dict = {}
        keys = data[0].keys()
        for k in keys:
            input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
        audio_embeds = self.encode_audio(input_dict, device=device)["embedding"]
        audio_embeds = self.audio_projection(audio_embeds)
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        return audio_embeds

            

    def audio_infer(self, audio, hopsize=None, device=None):
        """Forward one audio and produce the audio embedding

        Parameters
        ----------
        audio:  (audio_length)
            the time-domain audio input, notice that it must be only one input
        hopsize: int
            the overlap hopsize as the sliding window

        Returns
        ----------
        output_dict: {
            key: [n, (embedding_shape)] if "HTS-AT"
            or
            key: [(embedding_shape)] if "PANN"
        }
            the list of key values of the audio branch

        """

        assert not self.training, "the inference mode must be run at eval stage"
        output_dict = {}
        # PANN
        if self.audio_cfg.model_type == "PANN":
            audio_input = audio.unsqueeze(dim=0)
            output_dict[key] = self.encode_audio(audio_input, device=device)[key].squeeze(dim=0)
        elif self.audio_cfg.model_type == "HTSAT":
            # repeat
            audio_len = len(audio)
            k = self.audio_cfg.clip_samples // audio_len
            if k > 1:
                audio = audio.repeat(k)
                audio_len = len(audio)

            if hopsize is None:
                hopsize = min(hopsize, audio_len)

            if audio_len > self.audio_cfg.clip_samples:
                audio_input = [
                    audio[pos : pos + self.audio_cfg.clip_samples].clone()
                    for pos in range(
                        0, audio_len - self.audio_cfg.clip_samples, hopsize
                    )
                ]
                audio_input.append(audio[-self.audio_cfg.clip_samples :].clone())
                audio_input = torch.stack(audio_input)
                output_dict[key] = self.encode_audio(audio_input, device=device)[key]
            else:
                audio_input = audio.unsqueeze(dim=0)
                output_dict[key] = self.encode_audio(audio_input, device=device)[key].squeeze(dim=0)

        return output_dict


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


# Ignore the state dict of the vision part
def build_model_from_openai_state_dict(state_dict: dict, model_cfg, enable_fusion: bool = False, fusion_type: str = 'None'):

    embed_dim = model_cfg["embed_dim"]
    audio_cfg = model_cfg["audio_cfg"]
    text_cfg = model_cfg["text_cfg"]
    vocal_cfg = model_cfg["vocal_cfg"]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2]
            for k in state_dict
            if k.startswith(f"transformer.resblocks")
        )
    )

    audio_cfg = CLAPAudioCfp(**audio_cfg)
    text_cfg = CLAPTextCfg(**text_cfg)
    vocal_cfg = CLAPVocalCfg(**vocal_cfg)

    model = CLVAP(
        embed_dim,
        audio_cfg=audio_cfg,
        text_cfg=text_cfg,
        vocal_cfg=vocal_cfg,
        quick_gelu=True,  # OpenAI models were trained with QuickGELU
        enable_fusion=enable_fusion,
        fusion_type=fusion_type
    )
    state_dict["logit_scale_a"] = state_dict["logit_scale"]
    state_dict["logit_scale_t"] = state_dict["logit_scale"]
    pop_keys = list(state_dict.keys())[::]
    # pop the visual branch saved weights
    for key in pop_keys:
        if key.startswith("visual."):
            state_dict.pop(key, None)

    for key in ["logit_scale", "input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    # not use fp16
    # convert_weights_to_fp16(model)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device("cpu")):
    model.eval()
    audio_length = model.audio_cfg.audio_length
    example_audio = torch.ones((batch_size, audio_length), device=device)
    example_text = torch.zeros(
        (batch_size, model.context_length), dtype=torch.int, device=device
    )
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_audio, example_text),
            encode_text=(example_text,),
            encode_image=(example_audio,),
        ),
    )
    model.audio_cfg.audio_length = audio_length  # Question: what does this do?
    return model
