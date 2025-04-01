# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from skimage import io
import torch.nn.functional as F

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_SemanticKitti(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, data_config, stereo_path, dataset, is_train=False, img_norm_cfg=None):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.img_norm_cfg = img_norm_cfg
        self.stereo_path = stereo_path
        self.dataset = dataset

    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, depth, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        
        # adjust image

        img, depth = self.img_transform_core(img, depth, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, depth, post_rot, post_tran

    def img_transform_core(self, img, depth, resize_dims, crop, flip, rotate):
        
        original_depth_size = depth.shape[-2:]

        # 2. Resize both the image and depth to the same dimensions (using NEAREST for depth)
        img = img.resize(resize_dims)
        depth = TF.resize(depth, (resize_dims[1], resize_dims[0]), interpolation=TF.InterpolationMode.BILINEAR)
               
        img = img.crop(crop)
        depth = TF.crop(depth, crop[1], crop[0], crop[3] - crop[1], crop[2] - crop[0])
        
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
            depth = TF.hflip(depth)
    
        img = img.rotate(rotate)
        depth = TF.rotate(depth, angle=rotate, interpolation=TF.InterpolationMode.BILINEAR)

        # resize the depth back to its original size using NEAREST interpolation
        depth = TF.resize(depth, original_depth_size, interpolation=TF.InterpolationMode.NEAREST)
        
        return img, depth

    
    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        
        if self.is_train:
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        
        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        
        return resize, resize_dims, crop, flip, rotate
    
    def depth2prob(self, depths):
        depths = (depths - (2.0 - 0.5 / 2)) / 0.5
        
        # depths = torch.where((depths < self.D + 1) & (depths >= 0.0), depths, torch.zeros_like(depths))
        depths = torch.clip(depths, 0, 112)
        depths = F.one_hot(depths.long(), num_classes=112 + 1)[..., 1:]

        # to gausiaan
        sigma = 0.5
        classes = torch.arange(112, dtype=torch.float32).view(1, 1, 1, 1, -1).to(depths.device)
        hot_indices = depths.argmax(dim=-1, keepdim=True)
        gauss_map = torch.exp(-0.5 * ((classes - hot_indices.float()) / sigma) ** 2)
        gauss_map /= gauss_map.sum(dim=-1, keepdim=True)
        depths = gauss_map.permute(0,1,4,2,3).flatten(0,1)
        return depths.float().detach()


    def get_inputs(self, results, flip=None, scale=None):
        # load the monocular image for semantic kitti

        imgs = []
        rots = []
        trans = []
        rel_rots = []
        rel_trans = []
        intrins = []
        post_rots = []
        post_trans = []
        stereo_depths = []
        cam2lidars = []
        canvas = []


        img_filenames = results['img_filename']
        lidar2img = results['lidar2img']
        cam_intrinsic = results['cam_intrinsic']
        lidar2cam = results['lidar2cam']
        lidar2prevcam = results['current2previous']

        for i in range(len(img_filenames)):
                    
            if self.dataset == 'kitti':
                seq_id, _, filename = img_filenames[i].split("/")[-3:]
            elif self.dataset == 'bench':
                seq_id, _, _, filename = img_filenames[i].split("/")[-4:]
            depth_path = os.path.join(self.stereo_path,seq_id,'depth',filename.replace(".png", ".npy"))
            stereo_depth = torch.from_numpy(np.load(depth_path)).unsqueeze(0)
            
            # load image
            img = mmcv.imread(img_filenames[i], 'unchanged')
            img = Image.fromarray(img)

            # image view augmentation (resize, crop, horizontal flip, rotate)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            img_augs = self.sample_augmentation(H=img.height,
                                                W=img.width,
                                                flip=flip,
                                                scale=scale)

            resize, resize_dims, crop, flip, rotate = img_augs
            img, stereo_depth, post_rot2, post_tran2 = \
                self.img_transform(img, stereo_depth,  post_rot,
                                    post_tran,
                                    resize=resize,
                                    resize_dims=resize_dims,
                                    crop=crop,
                                    flip=flip,
                                    rotate=rotate)

            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            
            # intrins
            intrin = torch.Tensor(cam_intrinsic[i])
            
            # extrins
            sensor2lidar = torch.tensor(lidar2cam[i]).inverse().float()
            rot = sensor2lidar[:3, :3]
            tran = sensor2lidar[:3, 3]
            
            # relative extrins
            rel_sensor2lidar = torch.tensor(lidar2prevcam[i]).inverse().float() 
            rel_rot = rel_sensor2lidar[:3, :3]
            rel_tran = rel_sensor2lidar[:3, 3]

            # vizualization for debuging
            # img_vis = np.array(img)
            # depth_vis = gt_depth.clone()

            # depth_bins = torch.arange(2.0, 58.0, 0.5).view(-1,1,1)
            # depth_vis = torch.sum(depth_vis * depth_bins,dim=0)

            # depth_vis = np.round(np.array(depth_vis) * 256).astype(np.uint16)
            # io.imsave(os.path.join('debug',seq_id + filename+'depth.png'), depth_vis)

            # io.imsave(os.path.join('debug',seq_id + filename+'img.png'), img_vis)

            
            # append multi frames
            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img, img_norm_cfg=self.img_norm_cfg))
            stereo_depths.append(stereo_depth)
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            cam2lidars.append(sensor2lidar)
            rel_rots.append(rel_rot)
            rel_trans.append(rel_tran)

        imgs = torch.stack(imgs)
        rots = torch.stack(rots)
        trans = torch.stack(trans)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        stereo_depths = torch.stack(stereo_depths)
        cam2lidars = torch.stack(cam2lidars)
        rel_rots = torch.stack(rel_rots)
        rel_trans = torch.stack(rel_trans)

        # the RGB uint8 input images, for debug or visualization
        results['canvas'] = np.stack(canvas)

        stereo_depths = self.depth2prob(stereo_depths)

        return imgs, rots, trans, rel_rots, rel_trans, intrins, post_rots, post_trans, stereo_depths, cam2lidars, torch.zeros(1)


    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        
        return results
    

def mmlabNormalize(img, img_norm_cfg=None):
    from mmcv.image.photometric import imnormalize
    if img_norm_cfg is None:
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        to_rgb = True
    else:
        mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
        std = np.array(img_norm_cfg['std'], dtype=np.float32)
        to_rgb = img_norm_cfg['to_rgb']
    
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    
    return img