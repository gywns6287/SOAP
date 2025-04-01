
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)
import torch.nn.functional as F
# main class

class DeformableAttention3D(nn.Module):
    def __init__(
            self,
            embed_dims=256,
            num_heads = 8,
            num_points=5,
            dropout=0.,
            batch_first=True,
            ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_points = num_points
        self.num_heads = num_heads
        
        self.sampling_offsets = nn.Linear(
            embed_dims,  num_heads * num_points * 3)
        self.attention_weights = nn.Linear(embed_dims, 
                                           num_heads * num_points)
        self.value_proj = nn.Conv3d(embed_dims, embed_dims, 1)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin(), thetas*0], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            3).repeat(1, 1, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True


    def forward(self,
                query,
                value,
                pos=None,
                reference_points=None,
                **kwargs):
        """
        query: (bs, Nq, C) // bs: batch size, Nq: query number, C: embedding dimension
        value: (bs, C, X, Y, Z)
        pos: (bs, Nq, C)
        reference_points: (bs, Nq, 3): the last dimension includes (x, y, z) coordinates corresonding to values. 
        """

        bs,  num_query, embed_dims = query.shape
        _, _, X, Y, Z = value.shape
        if pos is not None:
            query = query + pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)

        # value projection
        value = self.value_proj(value)
        value = value.reshape(bs, self.num_heads, -1 , X, Y, Z)
        
        # sampling offests and weights projection
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_points, 3)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_points)
        attention_weights = attention_weights.softmax(-1).contiguous()

        # calculate sampling locations
        offset_normalizer = torch.tensor([X, Y, Z], device=value.device)
        sampling_locations = reference_points[:, :, None, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, None, :]
        # norm to -1 ~ 1
        sampling_locations = 2 * sampling_locations - 1
        # (x,y,z) coordinates to (z,y,x)
        sampling_locations = sampling_locations[...,[2,1,0]]

        # sampling
        # bs, head, c//head, x, y, z -> (bs * head), c//head, x, y, z
        value = value.flatten(0,1)
        # bs, Nq, head, point, 3 -> (bs * head), Nq, points, 1, 3
        sampling_locations = sampling_locations.transpose(1,2).flatten(0,1).unsqueeze(-2)
        sampling_value = F.grid_sample(value, sampling_locations, align_corners=False)
        # (bs * head), c, Nq, points, 1 -> bs, head, c, Nq, points
        sampling_value = sampling_value.squeeze(-1).view(bs, self.num_heads, -1, num_query, self.num_points)
        
        # weights sum
        # bs, Nq, head, point -> bs, head, 1, Nq, points
        attention_weights = attention_weights.transpose(1,2).unsqueeze(2)
        # bs, head, c//head, Nq, points -> bs, head, c//head, Nq 
        output = (sampling_value * attention_weights).sum(-1)
        # -> bs, c, Nq -> bs, Nq, c 
        output = output.flatten(1,2).transpose(1,2)
        output = self.output_proj(output)
        return output 