import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmdet3d.models.builder import BACKBONES
from mmcv.runner import BaseModule
from mmcv.cnn import constant_init, trunc_normal_init, caffe2_xavier_init
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
from mmcv.utils import TORCH_VERSION, digit_version
from .ops.modules import MSDeformAttn
import torch.nn.functional as F
import copy
from mmcv.cnn.bricks.transformer import build_positional_encoding
from projects.mmdet3d_plugin.utils.point_generator import MlvlPointGenerator
from ..necks.multi_scale_deform_attn_3d import MultiScaleDeformableAttention3D
from .modules import BottleNeckASPP, ShiftWindowMSA, SwinBlock
from einops import rearrange

@BACKBONES.register_module()
class VRModule(BaseModule):
    '''
    Region separable transformer
    '''
    def __init__(
            self,
            in_channels,
            num_stage=4,
            block_numbers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            block_strides=[1, 2, 2, 2],
            out_indices=(0, 1, 2, 3),
            norm_cfg=dict(type='BN3d', requires_grad=True),
            num_cams= 1,
            with_cp=True,
            **kwargs,
        ):
        
        super().__init__()
        self.out_indices = out_indices
        self.num_stage = num_stage
        self.block_numbers = block_numbers
        feat_channels = in_channels

        self.input_proj = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(num_stage):
            channels = block_inplanes[i]
            self.input_proj.append(nn.Sequential(
                nn.Conv3d(in_channels, channels, kernel_size=1, bias=False),
                nn.GELU(),
                build_norm_layer(norm_cfg, channels)[1],
                ))
            
            nhead = channels//32
      
            self.decoder.append(
                DeformableTransformerDecoder(num_layers=block_numbers[i],
                                            channels=channels,
                                            feat_channels=feat_channels,
                                            n_heads=nhead,
                                            stage_idx = i,
                                            num_cams = num_cams)
            )
            
            in_channels = block_inplanes[i]


        # with torch.checkpoint
        self.with_cp = with_cp
        positional_encoding=dict(
            type='SinePositionalEncoding3D',
            num_feats=in_channels // 3,
            normalize=True)
        self.decoder_positional_encoding = build_positional_encoding(positional_encoding)
        self.point_generator = MlvlPointGenerator([2, 4, 8, 16])
        self.block_inplanes = block_inplanes

    def forward(self, mlvl_voxel_features, geo_inputs, mlvl_masks):


        # to reverse order
        # mlvl_voxel_features = mlvl_voxel_features[::-1]
        # mlvl_masks = mlvl_masks[::-1]
        
        res = []
        for i, (voxel_features, voxel_mask) \
            in enumerate(zip( mlvl_voxel_features, mlvl_masks)):

            # input projection
            voxel_features = self.input_proj[i](voxel_features)
            bs, c, bev_h, bev_w, bev_z = voxel_features.shape

            # transformer block 
            voxel_features = voxel_features.permute(0,2,3,4,1).flatten(1,3)
                    
            voxel_features = self.decoder[i](voxel_features, voxel_mask)

            voxel_features = voxel_features.permute(0,2,1).view(bs, c, bev_h, bev_w, bev_z)
            res.append(voxel_features)
        
        return res
    
    def get_ref_3d(self, bev_sizes):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
            ref_3d (Array): 3D reference points
        """

        scene_size = (51.2, 51.2, 6.4)
        vox_origin = np.array([0, -25.6, -2])
        voxel_size = 51.2 / bev_sizes[0]

        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels index in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        idx = np.array([range(vol_dim[0]*vol_dim[1]*vol_dim[2])])
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        vox_coords = np.concatenate([xv.reshape(1,-1), yv.reshape(1,-1), zv.reshape(1,-1), idx], axis=0).astype(int).T

        # Normalize the voxels centroids in lidar cooridnates
        ref_3d = np.concatenate([(xv.reshape(1,-1)+0.5)/bev_sizes[0], (yv.reshape(1,-1)+0.5)/bev_sizes[1], (zv.reshape(1,-1)+0.5)/bev_sizes[2],], axis=0).astype(np.float64).T 

        return vox_coords, ref_3d
    
    @force_fp32(apply_to=('reference_points', 'geo_inputs'))
    def point_sampling(self, reference_points, geo_inputs):
        pc_range = [0, -25.6, -2.0, 51.2, 25.6, 4.4] # default pc_range
        img_shape = (384, 1280)
        B, C, N, _ = reference_points.shape
        
        (rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = geo_inputs
        points = reference_points.clone().to(torch.float32)
        points[..., 0:1] = points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        points[..., 1:2] = points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        points[..., 2:3] = points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]


        points = torch.cat((points, torch.ones(*points.shape[:-1], 1).type_as(points)), dim=-1)
        bda = bda.view(B,1,1,4,4)
        # 3d agumentation transform
        points = torch.inverse(bda).matmul(points.unsqueeze(-1)).squeeze(-1) # 5
        points = points[...,:3]

       
        # lidar -> cam coords
        points = points - trans.view(B, C, 1, 3)
        points = torch.inverse(rots.unsqueeze(2)).matmul(points.unsqueeze(-1))

        # cam 3d -> 2d coords
        shift = intrins[:, :, :3, 3]
        intrins = intrins[:, :, :3, :3]
        points = intrins.unsqueeze(2).matmul(points).squeeze(-1)
        points = points + shift.unsqueeze(2)
        
        eps = 1e-5
        bev_mask = (points[..., 2:3] > eps)
        points = points[..., 0:2] / torch.maximum(
            points[..., 2:3], torch.ones_like(points[..., 2:3]) * eps) # 2
        points = torch.cat((points, torch.ones(*points.shape[:-1], 1).type_as(points)), dim=-1)
       
        # 2d agumentation transform
        points = post_rots.unsqueeze(2).matmul(points.unsqueeze(-1)).squeeze(-1)
        points = points + post_trans.unsqueeze(2)

        # normalize to 1-0
        points = points[...,:2]
        points[..., 0] /= img_shape[1]
        points[..., 1] /= img_shape[0]

        bev_mask = (bev_mask & (points[..., 1:2] > 0.0)
                    & (points[..., 1:2] < 1.0)
                    & (points[..., 0:1] < 1.0)
                    & (points[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))
        return points, bev_mask.squeeze(-1)

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                feat_channels=128,
                dropout=0.1, activation="relu",
                n_levels=4, n_heads=8, n_points=4, shift=False, stage_idx=1):
        super().__init__()

        self.shift= shift
        # self attention
        self.self_attn = SwinBlock(
            embed_dims=d_model,
            num_heads=n_heads,
            feedforward_channels=d_model,
            window_size=7,
            drop_path_rate=0.2,
            shift=shift)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)


        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.vox_size = (128//2**stage_idx, 128//2**stage_idx, 16//2**stage_idx)
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, voxel_mask):

        # self attention, 
        # to voxel shape
        bs, _, c = tgt.shape
        tgt2 = tgt.permute(0,2,1).view(bs,c,self.vox_size[0],self.vox_size[1],self.vox_size[2])
        
        # only swin - for object features
        tgt2 = rearrange(tgt2, 'b c x y z -> (b z) c x y')
        voxel_mask = rearrange(voxel_mask, 'b x y z -> (b z) x y')
        tgt2 = self.self_attn(tgt2, voxel_mask)
        tgt2 = rearrange(tgt2, '(b z) c x y -> b c x y z', b=bs)
        voxel_mask = rearrange(voxel_mask, '(b z) x y -> b x y z', b=bs)
        # to query shape
        tgt2 = tgt2.permute(0,2,3,4,1).flatten(1,3)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, num_layers, channels, feat_channels, n_heads, stage_idx, num_cams = 1, return_intermediate=False):
        super().__init__()
        self.num_cams = num_cams
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.layers = self._make_layer(
            block = DeformableTransformerDecoderLayer,
            d_model = channels,
            d_ffn = channels,
            feat_channels = feat_channels,
            n_heads = n_heads,
            num_block = num_layers,
            stage_idx= stage_idx)



    def _make_layer(self, block, d_model, d_ffn, feat_channels, n_heads, num_block,stage_idx):

        layers = []
        for i in range(num_block):
            shift = (i % 2) == 1
            layers.append(
                block(d_model, d_ffn,
                    feat_channels,
                    activation='gelu',
                    dropout=0., 
                    n_levels=self.num_cams, n_heads=n_heads, n_points=4, shift=shift,
                    stage_idx= stage_idx))
        
        return nn.Sequential(*layers)



    def forward(self, tgt, voxel_mask):
        output = tgt

        intermediate = []
        for lid, layer in enumerate(self.layers):
            output = layer(output, voxel_mask)

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
