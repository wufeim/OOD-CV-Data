import copy
import os

import BboxTools as bbt
import numpy as np
import torch
import torchvision
from PIL import Image
import skimage
from torch.utils.data import Dataset

from src.models.utils import construct_class_by_name

CATEGORIES = [
    'aeroplane', 'bicycle', 'boat', 'bus',
    'car', 'chair', 'diningtable', 'motorbike',
    'sofa', 'train']


class Pascal3DPlus(Dataset):
    def __init__(
        self,
        data_type,
        category,
        root_path,
        transforms,
        occ_level=0,
        enable_cache=True,
        **kwargs,
    ):
        self.data_type = data_type
        self.root_path = root_path
        self.category = category
        self.occ_level = occ_level
        self.enable_cache = enable_cache
        self.transforms = torchvision.transforms.Compose(
            [construct_class_by_name(**t) for t in transforms]
        )

        if self.category == 'all':
            self.category = CATEGORIES
        if not isinstance(self.category, list):
            self.category = [self.category]
        self.multi_cate = len(self.category) > 1

        self.image_path = os.path.join(self.root_path, data_type, "images")
        self.annotation_path = os.path.join(self.root_path, data_type, "annotations")
        self.list_path = os.path.join(self.root_path, data_type, "lists")

        file_list = []
        self.subtypes = {}
        for cate in self.category:
            if self.occ_level == 0:
                _list_path = os.path.join(self.list_path, cate)
            else:
                _list_path = os.path.join(self.list_path, f"{cate}FGL{self.occ_level}_BGL{self.occ_level}")

            if cate not in self.subtypes:
                self.subtypes[cate] = [t.split(".")[0] for t in os.listdir(_list_path)]

            _file_list = sum(
                (
                    [
                        os.path.join(cate if self.occ_level == 0 else f"{cate}FGL{self.occ_level}_BGL{self.occ_level}", l.strip())
                        for l in open(
                            os.path.join(_list_path, subtype_ + ".txt")
                        ).readlines()
                    ]
                    for subtype_ in self.subtypes[cate]
                ),
                [],
            )
            file_list += [(f, cate) for f in _file_list]
        self.file_list = list(set(file_list))
        self.cache = {}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        name_img, cate = self.file_list[item]

        if self.enable_cache and name_img in self.cache.keys():
            sample = copy.deepcopy(self.cache[name_img])
        else:
            img = Image.open(os.path.join(self.image_path, f"{name_img}.JPEG"))
            if img.mode != "RGB":
                img = img.convert("RGB")
            annotation_file = np.load(
                os.path.join(self.annotation_path, name_img.split(".")[0] + ".npz"),
                allow_pickle=True,
            )

            this_name = name_img.split(".")[0]

            label = 0 if len(self.category) == 0 else self.category.index(cate)

            sample = {
                "this_name": this_name,
                "cad_index": int(annotation_file["cad_index"]),
                "azimuth": float(annotation_file["azimuth"]),
                "elevation": float(annotation_file["elevation"]),
                "theta": float(annotation_file["theta"]),
                "distance": 5.0,
                "bbox": annotation_file["box_obj"],
                "img": img,
                "original_img": np.array(img),
                "label": label,
            }

            if self.enable_cache:
                self.cache[name_img] = copy.deepcopy(sample)

        if self.transforms:
            sample = self.transforms(sample)

        return sample


class Resize:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.transform = torchvision.transforms.Resize(size=(height, width))

    def __call__(self, sample):
        assert len(sample['img'].shape) == 4
        b, c, h, w = sample['img'].shape
        if h != self.height or w != self.width:
            sample['img'] = self.transform(sample['img'])
            if 'kp' in sample:
                assert len(sample['kp'].shape) == 3
                sample['kp'][:, :, 0] *= self.width / w
                sample['kp'][:, :, 1] *= self.height / h
        return sample


class ToTensor:
    def __init__(self):
        self.trans = torchvision.transforms.ToTensor()

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])
        if "kpvis" in sample and not isinstance(sample["kpvis"], torch.Tensor):
            sample["kpvis"] = torch.Tensor(sample["kpvis"])
        if "kp" in sample and not isinstance(sample["kp"], torch.Tensor):
            sample["kp"] = torch.Tensor(sample["kp"])
        return sample


class Normalize:
    def __init__(self):
        self.trans = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])
        return sample
