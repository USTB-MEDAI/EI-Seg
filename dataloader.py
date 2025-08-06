from pathlib import Path
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    RandomSwap,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
from torchio.data import UniformSampler, LabelSampler
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
import torchio
from torchio import AFFINE, DATA
import torchio as tio
import torch
import sys

sys.path.append('./')


def get_subjects(conf):
    '''
    @description: get the subjects for normal training
    '''
    subjects = []
    if 'predict' in conf.job_name:
        img_path = Path(conf.pred_data_path)
        gt_path = Path(conf.pred_gt_path)
    else:
        img_path = Path(conf.data_path)
        gt_path = Path(conf.gt_path)
    x_generator = sorted(img_path.glob(f'*{conf.suffix}'))
    gt_generator = sorted(gt_path.glob(f'*{conf.suffix}'))
    
    if conf.debug_mode:
        x_generator = x_generator[:1]
        gt_generator = gt_generator[:1]
        
    for i, (source, gt) in enumerate(zip(x_generator, gt_generator)):
        subject = tio.Subject(
            source=tio.ScalarImage(source),
            gt=tio.LabelMap(gt),
        )
        subjects.append(subject)
    return subjects


class Dataset(torch.utils.data.Dataset):

    def __init__(self, conf):
        self.subjects = []

        queue_length = 10
        samples_per_volume = conf.samples_per_volume

        self.subjects = get_subjects(conf)

        self.transforms = self.transform(conf)

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)
        if conf.sampler_type == 'uniform':
            sampler = UniformSampler(patch_size=conf.patch_size)
        elif conf.sampler_type == 'label':
            sampler = LabelSampler(patch_size=conf.patch_size)

        self.queue_dataset = Queue(self.training_set, queue_length, samples_per_volume, sampler, num_workers=conf.num_workers)

    def transform(self, conf):

        if conf.aug:
            training_transform = Compose([
                # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                # RandomMotion(),
                # RandomBiasField(),
                ZNormalization(),
                RandomNoise(),
                RandomFlip(axes=(0, )),
                RandomAffine(degrees=15)
                # OneOf({
                #     RandomAffine(): 0.8,
                #     RandomElasticDeformation(): 0.2,
                # }),
            ])
            print('using aug!!')
        else:
            training_transform = Compose([
                # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                # RandomAffine(degrees=20),
                # RandomNoise(std=0.0001),
                ZNormalization(),
            ])
        return training_transform


def get_subjects_fold(conf, fold_total, fold_num):
    subjects_train = []
    subjects_val = []
    img_path = Path(conf.data_path)
    gt_path = Path(conf.gt_path)
    x_generator = sorted(img_path.glob(f'*{conf.suffix}'))
    gt_generator = sorted(gt_path.glob(f'*{conf.suffix}'))
    
    # if conf.debug_mode:
    #     x_generator = x_generator[:5]
    #     gt_generator = gt_generator[:5]
    
    Len = len(x_generator)
    val_l = Len // fold_total * (fold_num - 1)
    val_r = min(Len // fold_total * fold_num, Len)
    
    for i, (source, gt) in enumerate(zip(x_generator, gt_generator)):
        subject = tio.Subject(
            source=tio.ScalarImage(source),
            gt=tio.LabelMap(gt),
        )
        if i in range(val_l, val_r):
            subjects_val.append(subject)
        else:
            subjects_train.append(subject)
    return subjects_train, subjects_val, val_l, val_r


class Dataset_CycleVal(torch.utils.data.Dataset):

    def __init__(self, conf, fold_total, fold_num):

        queue_length = 10
        samples_per_volume = conf.samples_per_volume

        self.subjects_train, self.subjects_val, self.val_l, self.val_r = get_subjects_fold(conf, fold_total, fold_num)

        self.transforms = self.transform(conf)
        
        if conf.debug_mode:
            self.subjects_train = self.subjects_train[:1]

        self.training_set = tio.SubjectsDataset(self.subjects_train, transform=self.transforms)
        self.val_set = tio.SubjectsDataset(self.subjects_val, transform=self.transforms)


        if conf.sampler_type == 'uniform':
            sampler = UniformSampler(patch_size=conf.patch_size)
        elif conf.sampler_type == 'label':
            sampler = LabelSampler(patch_size=conf.patch_size)

        self.queue_dataset_train = Queue(self.training_set, queue_length, samples_per_volume, sampler, num_workers=conf.num_workers)
        self.queue_dataset_val = Queue(self.val_set, queue_length, samples_per_volume, sampler, num_workers=conf.num_workers)

    def transform(self, conf):

        if conf.aug:
            training_transform = Compose([
                # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                # RandomMotion(),
                # RandomBiasField(),
                ZNormalization(),
                RandomNoise(),
                RandomFlip(axes=(0, )),
                RandomAffine(degrees=15)
                # OneOf({
                #     RandomAffine(): 0.8,
                #     RandomElasticDeformation(): 0.2,
                # }),
            ])
            print('using aug!!')
        else:
            training_transform = Compose([
                # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                # RandomAffine(degrees=20),
                # RandomNoise(std=0.0001),
                ZNormalization(),
            ])
        return training_transform