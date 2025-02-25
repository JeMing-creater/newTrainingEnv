import os
import math
import yaml
import torch
import monai
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from typing import Optional
from easydict import EasyDict
from dataclasses import dataclass, field
from monai.utils import ensure_tuple_rep
from monai.networks.utils import one_hot
sitk.ProcessObject.SetGlobalWarningDisplay(False)
from torch.utils.data import Dataset, ConcatDataset
from typing import Tuple, List, Mapping, Hashable, Dict
from monai.transforms import (
    LoadImaged, MapTransform, ScaleIntensityRanged, EnsureChannelFirstd, Spacingd, Orientationd,ResampleToMatchd, ResizeWithPadOrCropd, Resize, Resized, RandFlipd, NormalizeIntensityd, ToTensord,RandScaleIntensityd,RandShiftIntensityd
)

@dataclass
class DataArguments:
    root_dir: str = field(
        default="/workspace/Jeming/data/",
        metadata={"help": "root_dir"},
    )
    check_models: Optional[List[str]] = field(
        default_factory=lambda: ["ADC", "T2_FS", "V"],
        metadata={"help": "check_models"},
    )
    model_scale: Optional[List[List[int]]] = field(
        default_factory=lambda: [[0,6000],[0,4500],[0,4500]],
        metadata={"help": "model_scale"},
    )
    target_size: Optional[List[int]] = field(
        default_factory=lambda: [128, 128, 64],
        metadata={"help": "check_models"},
    )
    train_ratio: Optional[float] = field(
        default = 0.7,
        metadata={"help": "train_ratio"},
    )
    val_ratio: Optional[float] = field(
        default = 0.1,
        metadata={"help": "val_ratio"},
    )
    test_ratio: Optional[float] = field(
        default = 0.2,
        metadata={"help": "test_ratio"},
    )
    
    
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(self, keys: monai.config.KeysCollection,
                 allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def converter(self, img: monai.config.NdarrayOrTensor):
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        result = [(img == 1)|(img == 3)]
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.concatenate(result, axis=0).astype(np.float32)

    def __call__(self, data: Mapping[Hashable, monai.config.NdarrayOrTensor]) -> Dict[
        Hashable, monai.config.NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


def load_MR_dataset_images(root, usedata, use_models):
    images_path = os.listdir(root)
    images_list = []
    for path in images_path:
        if path in usedata:
            models = os.listdir(root + '/' + path + '/')
            image = []
            label = []
            for model in models:
                if model in use_models:
                    image.append(root + '/' + path + '/' + model + '/' + path + '.nii.gz')
                    label.append(root + '/' + path + '/' + model + '/' + path + 'seg.nii.gz')
            images_list.append({
                'image': image,
                'label': label
            })
            
    return images_list


def read_usedata(file_path):
    read_flas = False
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if read_flas == True:
                result = line.replace('\n', '').split(',')
                result = [data.replace(' ', '') for data in result]
                return result
            elif 'Useful data' in line:
                read_flas = True
                continue


def get_GCM_transforms(args: DataArguments) -> Tuple[
    monai.transforms.Compose, monai.transforms.Compose, monai.transforms.Compose]:
    
    load_transform = []
    
    for model_scale in args.model_scale:
        load_transform.append(
            monai.transforms.Compose([
                LoadImaged(keys=["image", "label"], image_only=False, simple_keys=True),
                EnsureChannelFirstd(keys=["image", "label"]),
                Resized(keys=["image", "label"], spatial_size=args.target_size, mode=("trilinear", "nearest-exact")),
                
                ScaleIntensityRanged(
                        keys=["image"],  # 对图像应用变换
                        a_min=model_scale[0],  # 输入图像的最小强度值
                        a_max=model_scale[1],  # 输入图像的最大强度值
                        b_min=0.0,            # 输出图像的最小强度值
                        b_max=1.0,            # 输出图像的最大强度值
                        clip=True             # 是否裁剪超出范围的值
                    ),
                ToTensord(keys=['image', 'label'])
            ])
        )
    
    train_transform = monai.transforms.Compose([
        # 训练集的额外增强
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=['image', 'label'])
    ])
    val_transform = monai.transforms.Compose([
        ToTensord(keys=["image", "label"]),
    ])
    return load_transform, train_transform, val_transform


class GCMDataset(monai.data.Dataset):
    def __init__(self, data, loadforms, transforms):
        self.data = data
        self.transforms = transforms
        self.loadforms = loadforms
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        combined_data = {}
        
        for i in range(0, len(item['image'])):
            globals()[f'data_{i}'] = self.loadforms[i]({
                'image': item['image'][i],
                'label': item['label'][i]
            })

            combined_data[f'model_{i}_image'] = globals()[f'data_{i}']['image']
            combined_data[f'model_{i}_label'] = globals()[f'data_{i}']['label']
        
        images = []
        labels = []
        
        for i in range(0, len(item['image'])):
            images.append(combined_data[f'model_{i}_image'])
            labels.append(combined_data[f'model_{i}_label'])
            image_tensor = torch.cat(images, dim=0)
            label_tensor = torch.cat(labels, dim=0)
        
        result = {'image': image_tensor, 'label': label_tensor}
        result = self.transforms(result)
        return {'pixel_values': image_tensor, 'labels': label_tensor}    

def split_list(data, ratios):
    # 计算每个部分的大小
    sizes = [math.ceil(len(data) * r) for r in ratios]
    
    # 调整大小以确保总大小与原列表长度匹配
    total_size = sum(sizes)
    if total_size != len(data):
        sizes[-1] -= (total_size - len(data))
    
    # 分割列表
    start = 0
    parts = []
    for size in sizes:
        end = start + size
        parts.append(data[start:end])
        start = end
    
    return parts

def get_gcm_dataset(
        data_args: DataArguments,
    ) -> Tuple[Dataset, Dataset, Dataset]:   
    datapath = data_args.root_dir
    use_models = data_args.check_models
    # TODO: 统一数据名称
    datapath1 = datapath + '/' + 'NonsurgicalMR' + '/'
    datapath2 = datapath + '/' + 'SurgicalMR' + '/'
    usedata1 = datapath + '/' + 'NonsurgicalMR.txt'
    usedata2 = datapath + '/' + 'SurgicalMR.txt'
    usedata1 = read_usedata(usedata1)
    usedata2 = read_usedata(usedata2)
    
    data1 = load_MR_dataset_images(datapath1, usedata1, use_models)
    data2 = load_MR_dataset_images(datapath2, usedata2, use_models)
    
    data = data1 + data2
    
    load_transform, train_transform, val_transform = get_GCM_transforms(data_args)
    
    train_data, val_data, test_data = split_list(data, [data_args.train_ratio, data_args.val_ratio, data_args.test_ratio]) 

    train_dataset = GCMDataset( data=train_data, 
                                loadforms = load_transform,
                                transforms=train_transform)
    val_dataset   = GCMDataset( data=val_data, 
                                loadforms = load_transform,
                                transforms=val_transform)
    test_dataset  = GCMDataset( data=test_data, 
                                loadforms = load_transform,
                                transforms=val_transform)
    
    return train_dataset, val_dataset, test_dataset
    

if __name__ == '__main__':

    pass
    