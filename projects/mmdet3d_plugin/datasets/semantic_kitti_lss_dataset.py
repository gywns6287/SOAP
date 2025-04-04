import numpy as np
import glob
import os

from mmdet.datasets import DATASETS
from mmdet3d.datasets import SemanticKITTIDataset

from numpy.linalg import inv


@DATASETS.register_module()
class CustomSemanticKITTILssDataset(SemanticKITTIDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, split, camera_used, occ_size, pc_range, 
                 target_frames= [],load_continuous=False, *args, **kwargs):
        
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.camera_map = {'left': '2', 'right': '3'}
        self.camera_used = [self.camera_map[camera] for camera in camera_used]
        self.multi_scales = ["1_1", "1_2", "1_4", "1_8", "1_16"]
        
        self.load_continuous = load_continuous
        self.splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "trainval": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
            "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        }
        self.class_names = [
            'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
            'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
            'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
            'pole', 'traffic-sign'
        ]
        self.sequences = self.splits[split]
        self.n_classes = 20
        super().__init__(*args, **kwargs)
        self._set_group_flag()
        self.target_frames = target_frames
        self.poses = self.load_poses()

    @staticmethod
    def read_calib(calib_path):
        """calib.txt: Calibration data for the cameras: P0/P1 are the 3x4 projection
            matrices after rectification. Here P0 denotes the left and P1 denotes the
            right camera. Tr transforms a point from velodyne coordinates into the
            left rectified camera coordinate system. In order to map a point X from the
            velodyne scanner to a point x in the i'th image plane, you thus have to
            transform it like:
            x = Pi * Tr * X
            - 'image_00': left rectified grayscale image sequence
            - 'image_01': right rectified grayscale image sequence
            - 'image_02': left rectified color image sequence
            - 'image_03': right rectified color image sequence
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out["P2"] = np.identity(4)  # 4x4 matrix
        calib_out["P3"] = np.identity(4)  # 4x4 matrix
        calib_out["P2"][:3, :4] = calib_all["P2"].reshape(3, 4)
        calib_out["P3"][:3, :4] = calib_all["P3"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4) 
        
        return calib_out

    def load_annotations(self, ann_file=None):
        scans = []
        for sequence in self.sequences:
            calib = self.read_calib(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "calib.txt")
            )
            P2 = calib["P2"]
            P3 = calib["P3"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix_2 = P2 @ T_velo_2_cam
            proj_matrix_3 = P3 @ T_velo_2_cam

            voxel_base_path = os.path.join(self.ann_file, sequence)
            img_base_path = os.path.join(self.data_root, "dataset", "sequences", sequence)
                        
            if self.load_continuous:
                id_base_path = os.path.join(self.data_root, "dataset", "sequences", sequence, 'image_2', '*.png')
            else:
                id_base_path = os.path.join(self.data_root, "dataset", "sequences", sequence, 'voxels', '*.bin')
            
            for id_path in glob.glob(id_base_path):
                img_id = id_path.split("/")[-1].split(".")[0]
                img_2_path = os.path.join(img_base_path, 'image_2', img_id + '.png')
                img_3_path = os.path.join(img_base_path, 'image_3', img_id + '.png')
                voxel_path = os.path.join(voxel_base_path, img_id + '_1_1.npy')
                
                # for sweep demo or test submission
                if not os.path.exists(voxel_path):
                    voxel_path = None
                scans.append(
                    {   "img_2_path": img_2_path,
                        "img_3_path": img_3_path,
                        "sequence": sequence,
                        "frame_id": img_id,
                        "P2": P2,
                        "P3": P3,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix_2": proj_matrix_2,
                        "proj_matrix_3": proj_matrix_3,
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
        '''
        sample info includes the following:
            "img_2_path": img_2_path,
            "img_3_path": img_3_path,
            "sequence": sequence,
            "P2": P2,
            "P3": P3,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix_2": proj_matrix_2,
            "proj_matrix_3": proj_matrix_3,
            "voxel_path": voxel_path,
        '''
        
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
        filename = os.path.basename(img_path)
        frame_id = os.path.splitext(filename)[0]
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
                self.data_root, "dataset", "sequences", sequence, "image_2", target_id + ".png"
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
        

    def load_poses(self):
        """ read poses for each sequence

            Returns
            -------
            dict
                pose dict for different sequences.
        """
        pose_dict = dict()
        for sequence in self.sequences:
            pose_path = os.path.join(self.data_root, "dataset", "sequences", sequence, "poses.txt")
            calib = self.read_calib(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "calib.txt")
            )
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
        Tr_inv = inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses