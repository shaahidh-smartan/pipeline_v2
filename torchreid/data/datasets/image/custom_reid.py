from __future__ import division, print_function, absolute_import

import re
import glob
import os.path as osp

from ..dataset import ImageDataset


class CustomReID(ImageDataset):
    """Custom ReID dataset in Market1501 format"""
    dataset_dir = 'reid_dataset'
    masks_base_dir = 'masks'

    masks_dirs = {
        # dir_name: (parts_num, masks_stack_size, masks_suffix, parts_names)
        'yolo_pose_seg_filtering': (17, False, '.npy', ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                                                         'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                                                         'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                                                         'left_knee', 'right_knee', 'left_ankle', 'right_ankle']),
    }

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in CustomReID.masks_dirs:
            return None
        else:
            return CustomReID.masks_dirs[masks_dir]

    def __init__(self, root='', masks_dir=None, **kwargs):
        self.masks_dir = masks_dir
        if self.masks_dir in self.masks_dirs:
            mask_config = self.masks_dirs[self.masks_dir]
            self.masks_parts_numbers = mask_config[0]
            self.has_background = mask_config[1]
            self.masks_suffix = mask_config[2]
        else:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = None, None, None

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self.query_dir = osp.join(self.dataset_dir, 'query')

        gallery = self.process_dir(self.gallery_dir, relabel=False)
        query = self.process_dir(self.query_dir, relabel=False)

        super(CustomReID, self).__init__([], query, gallery, **kwargs)

    def infer_masks_path(self, img_path):
        """Infer masks path from image path"""
        if self.masks_dir is None or self.masks_suffix is None:
            return None

        # Convert: reid_dataset/gallery/0001_c1_000001.jpg
        #   to:    reid_dataset/masks/yolo_pose_seg_filtering/gallery/0001_c1_000001.jpg.npy
        relative_path = osp.relpath(img_path, self.dataset_dir)
        masks_path = osp.join(self.dataset_dir, self.masks_base_dir, self.masks_dir, relative_path + self.masks_suffix)
        return masks_path

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'(\d+)_c(\d+)')

        data = []
        for img_path in img_paths:
            match = pattern.search(img_path)
            if not match:
                continue

            pid, camid = map(int, match.groups())
            camid -= 1  # index starts from 0

            masks_path = self.infer_masks_path(img_path)

            data.append({
                'img_path': img_path,
                'pid': pid,
                'camid': camid,
                'masks_path': masks_path
            })

        return data
