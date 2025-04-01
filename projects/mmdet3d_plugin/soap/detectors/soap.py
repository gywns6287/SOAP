import torch
import torch.nn.functional as F
import collections 

from mmdet.models import DETECTORS
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.utils import fast_hist_crop
from .bevdepth import BEVDepth
import numpy as np
import time
import pdb
import os
from torch import nn
from mmdet3d.models import builder

@DETECTORS.register_module()
class SOAP(BEVDepth):
    def __init__(self, n_classes, **kwargs):
        super().__init__(**kwargs)
        
        self.record_time = False
        self.time_stats = collections.defaultdict(list)

        if type(kwargs['img_neck']['out_channels']) == list:
            featC = kwargs['img_neck']['out_channels'][0]
        else:
            featC = kwargs['img_neck']['out_channels']
        self.sem_dic_scores = nn.Parameter(torch.zeros(n_classes,10), requires_grad=False)
        self.sem_dic = nn.Parameter(torch.zeros(n_classes,10,featC), requires_grad=False)
        self.dic_embed = nn.Embedding(n_classes, featC)
        nn.init.xavier_uniform_(self.dic_embed.weight)
        self.update_for_infer = False

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        
        x = self.img_backbone(imgs)
        # use 1, 2, 3, 5 stage features
        mlvl_feats = self.img_neck(x)
        return mlvl_feats
    
    @force_fp32()
    def bev_encoder(self, mlvl_voxel_features, geo_inputs, mlvl_masks):
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        x = self.img_bev_encoder_backbone(mlvl_voxel_features, geo_inputs, mlvl_masks)
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['bev_encoder'].append(t1 - t0)
        

        x = self.img_bev_encoder_neck(x)
        
        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['bev_neck'].append(t2 - t1)
   
        return x
    
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        nframes = img[0].size(1)
        curr_img, prev_img = torch.split(img[0],[1,nframes-1],dim=1)
        
        # extract image features
        curr_mlvl_feats = self.image_encoder(curr_img)
        with torch.no_grad():
            prev_mlvl_feats = self.image_encoder(prev_img)
    
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['img_encoder'].append(t1 - t0)

        # img: imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors
        rots, trans, rel_rots, rel_trans, intrins, post_rots, post_trans, bda = img[1:9]

        # geometrical information -> features space
        geo_inputs_all = [rots, trans, intrins, post_rots, post_trans]
        curr_geo_inputs = [g[:,:1] for g in geo_inputs_all] + [bda]
        prev_geo_inputs = [g[:,1:] for g in geo_inputs_all] + [bda]
        curr_mlp_input = self.img_view_transformer.get_mlp_input(*curr_geo_inputs)
        with torch.no_grad():
            prev_mlp_input = self.img_view_transformer.get_mlp_input(*prev_geo_inputs)
        curr_geo_inputs = curr_geo_inputs[:2] + [rel_rots, rel_trans]\
                        + curr_geo_inputs[2:] + [curr_mlp_input]
        prev_geo_inputs = prev_geo_inputs[:2] + [rel_rots, rel_trans]\
                        + prev_geo_inputs[2:] + [prev_mlp_input]

        # Image-Voxel transformation 
        mlvl_voxel_features, mlvl_masks = self.img_view_transformer(img[9],
            curr_mlvl_feats, prev_mlvl_feats, curr_geo_inputs, prev_geo_inputs)

        # voxel encoder
        mlvl_voxel_features = self.bev_encoder(mlvl_voxel_features, curr_geo_inputs, mlvl_masks)

        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['view_transformer'].append(t2 - t1)
        
        return  mlvl_voxel_features



    def forward_dummy(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        voxel_feats = self.extract_feat(points=None, img=img_inputs, img_metas=img_metas)        
        output = self.pts_bbox_head.simple_test(
            voxel_feats=voxel_feats,
            points=None,
            img_metas=img_metas,
            img_feats=img_feats,
            points_uv=None,
        )
        
        bbox_list = [dict() for _ in range(1)]
        return bbox_list


    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        mlvl_voxel_features = self.extract_img_feat(img, img_metas)
        return mlvl_voxel_features
    
    @force_fp32(apply_to=('pts_feats'))
    def forward_pts_train(
            self,
            pts_feats,
            gt_occ=None,
            points_occ=None,
            img_metas=None,
            img_feats=None,
            points_uv=None,
            **kwargs,
        ):
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        losses, query_scores, query_feats = self.pts_bbox_head.forward_train(
            voxel_feats=pts_feats,
            img_metas=img_metas,
            gt_occ=gt_occ,
            points=points_occ,
            img_feats=img_feats,
            points_uv=points_uv,
            sem_dic = self.sem_dic.clone(), 
            sem_embed= self.dic_embed.weight,
            **kwargs,
        )

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['mask2former_head'].append(t1 - t0)
        
        return losses, query_scores, query_feats
    
    def forward_train(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            points_occ=None,
            points_uv=None,
            **kwargs,
        ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        
        # with torch.no_grad():
        # extract bird-eye-view features from perspective images
        voxel_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        # training losses
        losses = dict()
        losses_occupancy, query_scores, query_feats = self.forward_pts_train(voxel_feats, gt_occ, 
                        points_occ, img_metas, **kwargs)
        losses.update(losses_occupancy)
        
        query_scores = F.softmax(query_scores, dim=-1)[..., :-1]
        self.update_dict(query_feats, query_scores)
        if self.record_time:
            # logging latencies
            avg_time = {key: sum(val) / len(val) for key, val in self.time_stats.items()}
            sum_time = sum(list(avg_time.values()))
            out_res = ''
            for key, val in avg_time.items():
                out_res += '{}: {:.4f}, {:.1f}, '.format(key, val, val / sum_time)
                
            print(out_res)
        return losses
        
    def forward_test(self,
            img_metas=None,
            img_inputs=None,
            **kwargs,
        ):  

        return self.simple_test(img_metas, img_inputs, **kwargs)
    
    def simple_test(self, img_metas, img=None, rescale=False, points_occ=None, gt_occ=None, points_uv=None):
        voxel_feats = self.extract_feat(points=None, img=img, img_metas=img_metas)        
        
        output, query_scores, query_feats = self.pts_bbox_head.simple_test(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=None,
            points_uv=points_uv,
            sem_dic = self.sem_dic.clone(), 
            sem_embed = self.dic_embed.weight,
        )


        # evaluate voxel 
        output_voxels = output['output_voxels'][0]
        target_occ_size = img_metas[0]['occ_size']
        
        if (output_voxels.shape[-3:] != target_occ_size).any():
            output_voxels = F.interpolate(output_voxels, size=tuple(target_occ_size), 
                            mode='trilinear', align_corners=True)
        
        output['output_voxels'] = output_voxels
        output['target_voxels'] = gt_occ
        
        if self.update_for_infer:
            query_scores = F.softmax(query_scores, dim=-1)[..., :-1]
            self.update_dict(query_feats, query_scores)
        return output
    
    def update_dict(self, query, scores):
        scores = scores.clone().detach().flatten(0,1).T
        query = query.clone().detach().flatten(0,1)


        # novelty scores
        repo_feats = self.sem_dic.clone().mean(1)
        
        query_norm = F.normalize(query, dim=1)
        repo_norm = F.normalize(repo_feats, dim=1)
        novelty = -torch.mm(repo_norm, query_norm.T)
        query_scores = scores + 0.2*novelty
        metrics = torch.cat([query_scores, self.sem_dic_scores],dim=1)
        new_scores, topk_ind = torch.topk(metrics, k=10, dim=1)
        query = query.unsqueeze(0).expand(self.sem_dic.shape[0], -1, -1)  # (20, 100, C)
        temp_query = torch.cat([query, self.sem_dic], dim=1)
        # update semantic dictonary
        topk_ind2 = topk_ind.unsqueeze(-1).expand(-1, -1, query.shape[-1])
        updated_dict = torch.gather(temp_query, dim=1, index=topk_ind2)
        
        self.sem_dic.copy_(updated_dict.detach())
        self.sem_dic_scores.copy_(new_scores.detach())