# Copyright (c) Phigent Robotics. All rights reserved.
import torch
from torch import nn
from mmdet3d.models.builder import NECKS
from mmdet3d.ops.bev_pool import bev_pool
from mmcv.runner import force_fp32
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_positional_encoding
from .deform3d import DeformableAttention3D
from mmdet.models.backbones.resnet import BasicBlock
import pdb
from mmcv.runner import BaseModule
from .ViewTransformerLSSBEVDepth import DepthNet
import numpy as np

@NECKS.register_module()
class OAProjection(BaseModule):
    def __init__(
            self, 
            grid_config,
            mlvl_nums,
            data_config, 
            numC,
            cam_channels,
            **kwargs
        ):
        super(OAProjection, self).__init__(**kwargs)
        

        positional_encoding=dict(
            type='SinePositionalEncoding3D',
            num_feats=numC // 3,
            normalize=True)
        self.grid_config = grid_config
        self.data_config = data_config
        self.D = int((self.grid_config['dbound'][1] - self.grid_config['dbound'][0])//self.grid_config['dbound'][2])
        
        
        self.numC_Trans = numC
        self.mlvl_nums = mlvl_nums

        self.depth_net = nn.ModuleList()
        self.occ_cross_attn = nn.ModuleList()
        self.default_geom = []
        for l in range(mlvl_nums):
            self.default_geom.append(self.set_geometry(4*2**(l)))
            self.depth_net.append(
                DepthNet(numC, numC, numC, self.D, cam_channels=cam_channels)
            )
            self.occ_cross_attn.append(
                DeformableAttention3D(numC, numC//32)
            )
        self.occ_positional_encoding = build_positional_encoding(positional_encoding)
        

    def voxel_pooling(self, geom_feats, x, dx_bx_nx):
        #geom_feats = (x,y,z,3) coords of x (B, N, D, H, W, 3)

        dx, bx, nx = dx_bx_nx
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (bx - dx / 2.)) / dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        
        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        if torch.sum(kept) != 0:
            final = bev_pool(x, geom_feats, B, nx[2], nx[0], nx[1])
            # [b, c, z, x, y] == [b, c, x, y, z]
            final = final.permute(0, 1, 3, 4, 2)
        else:
            final = torch.zeros(
                B,C,int(nx[0]),int(nx[1]),int(nx[2])
                ).float().to(x.device)

        return final

    def get_geometry(self, frustum, rots, trans, intrins, post_rots, post_trans, bda):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape
        # undo post-transformation
        # B x N x D x H x W x 3    

        points = frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        
        if intrins.shape[3] == 4: # for KITTI
            shift = intrins[:, :, :3, 3]
            points = points - shift.view(B, N, 1, 1, 1, 3, 1)
            intrins = intrins[:, :, :3, :3]
        
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        
        if bda.shape[-1] == 4:
            points = torch.cat((points, torch.ones(*points.shape[:-1], 1).type_as(points)), dim=-1)
            points = bda.view(B, 1, 1, 1, 1, 4, 4).matmul(points.unsqueeze(-1)).squeeze(-1)
            
            points = points[..., :3]
            

        else:
            points = bda.view(B, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)


        return points


    def forward(self, depth_bins, curr_mlvl_feats, prev_mlvl_feats, curr_geos, prev_geos):
        
        (rots, trans, rel_rots, rel_trans, intrins, post_rots, post_trans, bda, mlp_input) = curr_geos[:9]

        curr_mlp_input = curr_geos[-1]
        prev_mlp_input = prev_geos[-1]
        N = prev_mlvl_feats[0].size(0)
        curr_rel_rots, prev_rel_rots = torch.split(rel_rots,[1,N],dim=1)
        curr_rel_trans, prev_rel_trans = torch.split(rel_trans,[1,N],dim=1)
        p_intrins, p_post_rots, p_post_trans = prev_geos[4:7]

        mlvl_voxel_features = []
        mlvl_masks = []
        for i, (curr_feats, prev_feats) \
        in enumerate(zip(curr_mlvl_feats, prev_mlvl_feats)):
            
            B, C, H, W = curr_feats.shape

            # probs
            with torch.no_grad():
                depths = depth_bins[0].clone() # for only one batch
                depths = F.interpolate(depths, [H,W], mode='bilinear')
                curr_prob, prev_prob = torch.split(depths,[1,N],dim=0)

            # Occlusion map
            occ_prob = torch.zeros_like(curr_prob).detach() # B, self.D, H, W
            occ_inds = torch.argmax(curr_prob,dim=1) 
            d_range = torch.arange(curr_prob.shape[1], device=curr_prob.device).view(1, -1, 1, 1)
            occ_prob = (d_range <= occ_inds.unsqueeze(1)).float()
            occ_prob = occ_prob.unsqueeze(1).unsqueeze(-1)
            
            # Lift
            curr_feat = self.depth_net[i](curr_feats, curr_mlp_input)
            curr_volume = curr_prob.unsqueeze(1) * curr_feat.unsqueeze(2)
            curr_volume = curr_volume.view(B, 1, -1, self.D, H, W)
            curr_volume = curr_volume.permute(0, 1, 3, 4, 5, 2) # P

            # for previous frame
            with torch.no_grad():
                prev_feat =  self.depth_net[i](prev_feats, prev_mlp_input)
                prev_volume = prev_prob.unsqueeze(1) * prev_feat.unsqueeze(2)
                prev_volume = prev_volume.view(B, N, -1, self.D, H, W)
                prev_volume = prev_volume.permute(0, 1, 3, 4, 5, 2) # P
         
            # Splat
            curr_geom = self.get_geometry(
                self.default_geom[i][0].to(curr_feat.device), curr_rel_rots, curr_rel_trans, intrins, post_rots, post_trans, bda
                )
            prev_geom = self.get_geometry(
                self.default_geom[i][0].to(curr_feat.device), prev_rel_rots, prev_rel_trans, p_intrins, p_post_rots, p_post_trans, bda
                )
            dx_bx_nx = [xs.to(curr_feat.device) for xs in self.default_geom[i][1:]]



            
            curr_feat = self.voxel_pooling(curr_geom, curr_volume, dx_bx_nx)
            prev_feat = self.voxel_pooling(prev_geom, prev_volume, dx_bx_nx)
            occ_region = self.voxel_pooling(curr_geom , occ_prob, dx_bx_nx)
            
            # Find invisible mask
            inv_mask = curr_feat.detach()
            inv_mask = (inv_mask.sum(1,keepdim=True) == 0)
            inv_mask = inv_mask.squeeze(0).squeeze(0)

            # find occlusion mask
            occ_mask = (occ_region == 0) & ~inv_mask
            occ_mask = occ_mask.squeeze(0).squeeze(0)

            B, _, X, Y, Z = curr_feat.shape

            # fill invisible region
            curr_feat[:,:,inv_mask] += prev_feat[:,:,inv_mask]
            
            # select most far voxel when there are no occluded regions
            if torch.sum(occ_mask) == 0:
                occ_mask[X//2,Y//2,Z-1] = 1

            # fill occ region
            occ_feat = curr_feat[:,:,occ_mask]
            pos_3d = self.occ_positional_encoding(torch.zeros(B,X,Y,Z)).float().to(curr_feat.device)
            ref_3d = self.get_ref_3d([X,Y,Z], curr_feat.device)

            # B, C, L -> B, L, C
            occ_feat = occ_feat.permute(0,2,1)
            occ_ref_3d = ref_3d[occ_mask.flatten(),:].unsqueeze(0)
            occ_pos = pos_3d[:,:,occ_mask].permute(0,2,1)

            tgt = self.occ_cross_attn[i](
                query = occ_feat,
                value = prev_feat,
                pos = occ_pos,
                reference_points=occ_ref_3d,
            )

            curr_feat[:,:,occ_mask] = (occ_feat + tgt).permute(0,2,1)
            

            # Find empty mask
            voxel_mask = curr_feat.detach()
            voxel_mask = (voxel_mask.sum(1) == 0).to(torch.uint8)

            mlvl_voxel_features.append(curr_feat)
            mlvl_masks.append(voxel_mask)
    
        return mlvl_voxel_features, mlvl_masks

    def set_geometry(self, img_downsize):
        # h, w = image_size
        # raw_w = self.data_config['input_size'][1]
        # img_downsize = raw_w//w
        voxel_downsize = img_downsize//2

        # default xbound
        xbound = self.grid_config['xbound'][:2] + [self.grid_config['xbound'][-1]*voxel_downsize] 
        ybound = self.grid_config['ybound'][:2] + [self.grid_config['ybound'][-1]*voxel_downsize] 
        zbound = self.grid_config['zbound'][:2] + [self.grid_config['zbound'][-1]*voxel_downsize] 

        dx, bx, nx = gen_dx_bx(xbound,  ybound, zbound)

        frustum = self.create_frustum(img_downsize)
        return frustum, dx, bx, nx
        
    
    def create_frustum(self, downsample):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        fH, fW = ogfH // downsample, ogfW // downsample
        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return frustum
    
    def get_ref_3d(self, bev_sizes, device):
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
        ref_3d = torch.from_numpy(ref_3d).float().to(device)
        return ref_3d

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda=None):
        B, N, _, _ = rot.shape
        
        if bda is None:
            bda = torch.eye(3).to(rot).view(1, 3, 3).repeat(B, 1, 1)
        
        bda = bda.view(B, 1, *bda.shape[-2:]).repeat(1, N, 1, 1)
        
        if intrin.shape[-1] == 4:
            # for KITTI, the intrin matrix is 3x4
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                intrin[:, :, 0, 3],
                intrin[:, :, 1, 3],
                intrin[:, :, 2, 3],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],
                bda[:, :, 0, 0],
                bda[:, :, 0, 1],
                bda[:, :, 1, 0],
                bda[:, :, 1, 1],
                bda[:, :, 2, 2],
            ], dim=-1)
            
            if bda.shape[-1] == 4:
                mlp_input = torch.cat((mlp_input, bda[:, :, :3, -1]), dim=2)
        else:
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],
                bda[:, :, 0, 0],
                bda[:, :, 0, 1],
                bda[:, :, 1, 0],
                bda[:, :, 1, 1],
                bda[:, :, 2, 2],
            ], dim=-1)
        
        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)], dim=-1).reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        
        return mlp_input
def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx