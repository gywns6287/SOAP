from math import ceil
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from ..backbones.modules import SwinBlock
from ..necks.multi_scale_deform_attn_3d import MultiScaleDeformableAttention3D
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from projects.mmdet3d_plugin.utils.point_generator import MlvlPointGenerator
from einops import rearrange

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu"):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = _get_activation_fn(activation)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, tgt):
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return tgt


class QueryFuse(nn.Module):

    def __init__(
        self,
        in_channels,
    ):

        super().__init__()
        nhead = in_channels//16
        self.cross_attn = nn.MultiheadAttention(in_channels, nhead)
        self.scale = nn.Parameter(1e-6 * torch.ones(1, 1, in_channels), requires_grad=True)
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query_feats, query_embed, semantic_dict, sem_embed):
        """

        """
        S, K, C = semantic_dict.shape
        sem_embed = sem_embed.unsqueeze(1).repeat(1,K,1)

        # encoding dictionary
        semantic_dict = semantic_dict.flatten(0,1).unsqueeze(1)
        sem_embed = sem_embed.flatten(0,1).unsqueeze(1)

        # query <-> repository
        query_feats2 = self.cross_attn(
            query=self.with_pos_embed(query_feats, query_embed),
            key=self.with_pos_embed(semantic_dict, sem_embed),
            value=semantic_dict
        )[0]
        query_feats = query_feats + query_feats2 * self.scale

        return query_feats       


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
