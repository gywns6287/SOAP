import numpy as np
import glob
import os

from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset

from numpy.linalg import inv
import yaml

@DATASETS.register_module()
class CustomSSCBenchLssDataset(Custom3DDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, 
                split, 
                data_root,
                ann_file,
                classes,
                camera_used, 
                occ_size, 
                pc_range, 
                pipeline=None,
                test_mode=False,
                load_continuous=False, 
                target_frames= [], 
                *args, **kwargs):
        
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.camera_map = {'left': '2', 'right': '3'}
        self.camera_used = [self.camera_map[camera] for camera in camera_used]
        self.multi_scales = ["1_1", "1_2", "1_4", "1_8", "1_16"]
        
        self.load_continuous = load_continuous
        self.splits = {
            "train": [ "2013_05_28_drive_0004_sync", "2013_05_28_drive_0000_sync", "2013_05_28_drive_0010_sync","2013_05_28_drive_0002_sync", "2013_05_28_drive_0003_sync", "2013_05_28_drive_0005_sync", "2013_05_28_drive_0007_sync"],
            "val": ["2013_05_28_drive_0006_sync"],
            "test": ["2013_05_28_drive_0009_sync"],
        }
        self.class_names = classes


        self.sequences = self.splits[split]
        self.n_classes = len(self.class_names)
        self.label_root = os.path.join(ann_file,'labels')
        self.calib = self.read_calib()
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            test_mode=test_mode)
        
        self.target_frames = target_frames
        self._set_group_flag()
        self.poses = self.load_poses(self.calib)

    @staticmethod
    def read_calib():

        P = np.array(
                [
                    552.554261,
                    0.000000,
                    682.049453,
                    0.000000,
                    0.000000,
                    552.554261,
                    238.769549,
                    0.000000,
                    0.000000,
                    0.000000,
                    1.000000,
                    0.000000,
                ]
            ).reshape(3, 4)
        # reshape matrices

        cam2velo = np.array(
                [   
            0.04307104361,
            -0.08829286498,
            0.995162929,
            0.8043914418,
            -0.999004371,
            0.007784614041,
            0.04392796942,
            0.2993489574,
            -0.01162548558,
            -0.9960641394,
            -0.08786966659,
            -0.1770225824,
                ]
        ).reshape(3, 4)
        C2V = np.concatenate(
            [cam2velo, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0
        )
        V2C = np.linalg.inv(C2V)
        # print("V2C: ", V2C)
        V2C = V2C[:3, :]

        cam_to_gps = np.array(
            [0.0371783278,
            -0.0986182135,
            0.9944306009,
            1.5752681039,
            0.9992675562,
            -0.0053553387, 
            -0.0378902567,
            0.0043914093,
            0.0090621821,
            0.9951109327,
            0.0983468786,
            -0.6500000000 ]
        ).reshape(3, 4)
        C2G =  np.concatenate(
            [cam_to_gps, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0
        )

        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = P
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = V2C
        calib_out['C2G'] = C2G
        return calib_out

    def load_annotations(self, ann_file=None):
        scans = []
        for sequence in self.sequences:
            calib = self.calib 

            P2 = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix_2 = P2 @ T_velo_2_cam

            voxel_base_path = os.path.join(ann_file, "labels", sequence)
            img_base_path = os.path.join(self.data_root,  "data_2d_raw", sequence, "image_00/data_rect")
                               
            if self.load_continuous:
                id_base_path = os.path.join(self.data_root,  "data_2d_raw", sequence, "image_00/data_rect", '*.png')
            else:
                id_base_path = os.path.join(self.data_root,  "data_2d_raw", sequence, 'voxels', '*.bin')

            for id_path in glob.glob(id_base_path):
                img_id = id_path.split("/")[-1].split(".")[0]
                img_2_path = os.path.join(img_base_path, img_id + '.png')
                voxel_path = os.path.join(voxel_base_path, img_id + '_1_1.npy')
                
                # for sweep demo or test submission
                if not os.path.exists(voxel_path):
                    voxel_path = None
                scans.append(
                    {   "img_2_path": img_2_path,
                        "sequence": sequence,
                        "frame_id": img_id,
                        "P2": P2,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix_2": proj_matrix_2,
                        "voxel_path": voxel_path,
                    })

        # return scans[:10] # for debuging        
        return scans  # return to self.data_infos

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in training data')
            return None
        
        # init for pipeline
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        
        return example

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data

    def get_ann_info(self, index):
        info = self.data_infos[index]['voxel_path']
        return [] if info is None else np.load(info)

    def get_data_info(self, index):
        info = self.data_infos[index]

        input_dict = dict(
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
            sequence = info['sequence'],
            frame_id = info['frame_id'],
        )
        
        # load images, intrins, extrins, voxels
        image_paths = []
        lidar2cam_rts = []
        lidar2img_rts = []
        cam_intrinsics = []
        curr2prev_rts = []

        # camera instrinsic
        P  = info['P2']

        # for cam_type in self.camera_used:
        img_path = info['img_2_path']
        image_paths.append(img_path)
        lidar2img_rts.append(info['proj_matrix_2'])
        cam_intrinsics.append(P)
        lidar2cam_rts.append(info['T_velo_2_cam'])
        curr2prev_rts.append(info['T_velo_2_cam'])

        # for prev frames
        frame_id = info['frame_id']
        sequence = info['sequence']
        pose_list = self.poses[sequence]
        seq_len = len(pose_list)
        
        for i in self.target_frames:
            id = int(frame_id)

            if id + i < 0 or id + i > seq_len-1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)
            
            img_path = os.path.join(
                self.data_root,  "data_2d_raw", sequence, "image_00/data_rect", target_id + ".png"
            )
            image_paths.append(img_path)
            lidar2img_rts.append(info['proj_matrix_2'])
            lidar2cam_rts.append(info['T_velo_2_cam'])
            cam_intrinsics.append(P)

            # current -> previous -> camera
            # cuurent lidar (ref) -> previous (target) lidar
            curr = pose_list[int(frame_id)] # reference frame with GT semantic voxel
            prev = pose_list[int(target_id)]
            curr2prev = np.matmul(inv(prev), curr) # both for lidar
            curr2cam = info["T_velo_2_cam"] 
            curr2prevcam = curr2cam @ curr2prev

            curr2prev_rts.append(curr2prevcam)
            



        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                current2previous=curr2prev_rts,
            ))
        
        # gt_occ is None for test-set
        input_dict['gt_occ'] = self.get_ann_info(index)

        return input_dict

    def evaluate(self, results, logger=None, **kwargs):
        if results is None:
            logger.info('Skip Evaluation')
        
        if 'ssc_scores' in results:
            # for single-GPU inference
            ssc_scores = results['ssc_scores']
            class_ssc_iou = ssc_scores['iou_ssc'].tolist()
            res_dic = {
                "SC_Precision": ssc_scores['precision'].item(),
                "SC_Recall": ssc_scores['recall'].item(),
                "SC_IoU": ssc_scores['iou'],
                "SSC_mIoU": ssc_scores['iou_ssc_mean'],
            }
        else:
            # for multi-GPU inference
            assert 'ssc_results' in results
            ssc_results = results['ssc_results']
            completion_tp = sum([x[0] for x in ssc_results])
            completion_fp = sum([x[1] for x in ssc_results])
            completion_fn = sum([x[2] for x in ssc_results])
            
            tps = sum([x[3] for x in ssc_results])
            fps = sum([x[4] for x in ssc_results])
            fns = sum([x[5] for x in ssc_results])
            
            precision = completion_tp / (completion_tp + completion_fp)
            recall = completion_tp / (completion_tp + completion_fn)
            iou = completion_tp / \
                    (completion_tp + completion_fp + completion_fn)
            iou_ssc = tps / (tps + fps + fns + 1e-5)
            
            class_ssc_iou = iou_ssc.tolist()
            res_dic = {
                "SC_Precision": precision,
                "SC_Recall": recall,
                "SC_IoU": iou,
                "SSC_mIoU": iou_ssc[1:].mean(),
            }
        
        class_names = self.class_names
        for name, iou in zip(class_names, class_ssc_iou):
            res_dic["SSC_{}_IoU".format(name)] = iou
        
        eval_results = {}
        for key, val in res_dic.items():
            eval_results['semkitti_{}'.format(key)] = round(val * 100, 2)
        
        eval_results['semkitti_combined_IoU'] = eval_results['semkitti_SC_IoU'] + eval_results['semkitti_SSC_mIoU']
        
        if logger is not None:
            logger.info('SemanticKITTI SSC Evaluation')
            logger.info(eval_results)
        
        return eval_results
        

    def load_poses(self,calib):
        """ read poses for each sequence

            Returns
            -------
            dict
                pose dict for different sequences.
        """
        pose_dict = dict()
        for sequence in self.sequences:
            pose_path = os.path.join(self.data_root, "data_2d_raw", sequence, "poses.txt")
            pose_dict[sequence] = self.parse_poses(pose_path, calib)
        return pose_dict
    

    @staticmethod
    def parse_poses(filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        C2G = calibration['C2G']
        V2G = np.matmul(C2G, Tr)
        Tr_inv = inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(inv(V2G), np.matmul(pose, V2G)))
            # poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        return poses
    
def unpack(compressed):
  ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
  uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
  uncompressed[::8] = compressed[:] >> 7 & 1
  uncompressed[1::8] = compressed[:] >> 6 & 1
  uncompressed[2::8] = compressed[:] >> 5 & 1
  uncompressed[3::8] = compressed[:] >> 4 & 1
  uncompressed[4::8] = compressed[:] >> 3 & 1
  uncompressed[5::8] = compressed[:] >> 2 & 1
  uncompressed[6::8] = compressed[:] >> 1 & 1
  uncompressed[7::8] = compressed[:] & 1

  return uncompressed