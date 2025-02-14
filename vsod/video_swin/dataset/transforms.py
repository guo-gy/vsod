import random

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils import data
import numpy as np
from torchvision.transforms import InterpolationMode


def get_transforms(input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return transforms.Compose(
        [Resize(input_size), ToTensor(), Normalize(mean=mean, std=std)]
    )


def get_train_transforms(
    input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    return transforms.Compose(
        [
            Resize(input_size),
            RandomFlip(),
            Random_crop(15),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )


class RandomFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability."""

    def __call__(self, samples):
        rand_flip_index = random.randint(0, 1)

        if rand_flip_index == 1:
            for i in range(len(samples)):
                sample = samples[i]

                image, gt, mask, edge, edge_distance, pl, prob, grey = (
                    sample["image"],
                    sample["gt"],
                    sample["mask"],
                    sample["edge"],
                    sample["edge_distance"],
                    sample["pl"],
                    sample["prob"],
                    sample["grey"],
                )

                image = F.hflip(image)
                gt = F.hflip(gt)
                mask = F.hflip(mask)
                edge = F.hflip(edge)
                edge_distance = F.hflip(edge_distance)
                pl = F.hflip(pl)
                prob = F.hflip(prob)
                grey = F.hflip(grey)

                (
                    sample["image"],
                    sample["gt"],
                    sample["mask"],
                    sample["edge"],
                    sample["edge_distance"],
                    sample["pl"],
                    sample["prob"],
                    sample["grey"],
                ) = (image, gt, mask, edge, edge_distance, pl, prob, grey)

                samples[i] = sample

        else:
            pass

        return samples


class Resize(object):
    """Resize PIL image use both for training and inference"""

    def __init__(self, size):
        self.size = size

    def __call__(self, samples):
        for i in range(len(samples)):
            sample = samples[i]
            image, gt, mask, edge, edge_distance, pl, prob, grey = (
                sample["image"],
                sample["gt"],
                sample["mask"],
                sample["edge"],
                sample["edge_distance"],
                sample["pl"],
                sample["prob"],
                sample["grey"],
            )

            image = F.resize(image, self.size, InterpolationMode.NEAREST)

            if gt is not None:
                gt = F.resize(gt, self.size, InterpolationMode.NEAREST)
                mask = F.resize(mask, self.size, InterpolationMode.NEAREST)
                edge = F.resize(edge, self.size, InterpolationMode.NEAREST)
                edge_distance = F.resize(
                    edge_distance, self.size, InterpolationMode.BILINEAR
                )
                pl = F.resize(pl, self.size, InterpolationMode.NEAREST)
                prob = F.resize(prob, self.size, InterpolationMode.BILINEAR)
                grey = F.resize(grey, self.size, InterpolationMode.BILINEAR)

            (
                sample["image"],
                sample["gt"],
                sample["mask"],
                sample["edge"],
                sample["edge_distance"],
                sample["pl"],
                sample["prob"],
                sample["grey"],
            ) = (image, gt, mask, edge, edge_distance, pl, prob, grey)

            samples[i] = sample

        return samples


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, samples):
        for i in range(len(samples)):
            sample = samples[i]
            image, gt, mask, edge, edge_distance, pl, prob, grey = (
                sample["image"],
                sample["gt"],
                sample["mask"],
                sample["edge"],
                sample["edge_distance"],
                sample["pl"],
                sample["prob"],
                sample["grey"],
            )

            image = F.to_tensor(image)

            if gt is not None:
                gt = F.to_tensor(gt)
                mask = F.to_tensor(mask)
                edge = F.to_tensor(edge)
                edge_distance = F.to_tensor(edge_distance)
                pl = F.to_tensor(pl)
                prob = F.to_tensor(prob)
                grey = F.to_tensor(grey)

            (
                sample["image"],
                sample["gt"],
                sample["mask"],
                sample["edge"],
                sample["edge_distance"],
                sample["pl"],
                sample["prob"],
                sample["grey"],
            ) = (image, gt, mask, edge, edge_distance, pl, prob, grey)

            samples[i] = sample

        return samples


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    args:    tensor (Tensor) ? Tensor image of size (C, H, W) to be normalized.
    Returns: Normalized Tensor image.
    """

    # default caffe mode
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, samples):
        for i in range(len(samples)):
            sample = samples[i]

            image, gt, mask, edge, edge_distance, pl, prob, grey = (
                sample["image"],
                sample["gt"],
                sample["mask"],
                sample["edge"],
                sample["edge_distance"],
                sample["pl"],
                sample["prob"],
                sample["grey"],
            )

            image = F.normalize(image, self.mean, self.std)

            (
                sample["image"],
                sample["gt"],
                sample["mask"],
                sample["edge"],
                sample["edge_distance"],
                sample["pl"],
                sample["prob"],
                sample["grey"],
            ) = (image, gt, mask, edge, edge_distance, pl, prob, grey)

            samples[i] = sample

        return samples


class Random_crop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, samples):
        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)

        width, height = samples[0]["image"].size
        assert samples[0]["image"].size == samples[0]["gt"].size
        region = [x, y, width - x, height - y]

        for i in range(len(samples)):
            sample = samples[i]

            image, gt, mask, edge, edge_distance, pl, prob, grey = (
                sample["image"],
                sample["gt"],
                sample["mask"],
                sample["edge"],
                sample["edge_distance"],
                sample["pl"],
                sample["prob"],
                sample["grey"],
            )

            image = image.crop(region)
            gt = gt.crop(region)
            mask = mask.crop(region)
            edge = edge.crop(region)
            edge_distance = edge_distance.crop(region)
            pl = pl.crop(region)
            prob = prob.crop(region)
            grey = grey.crop(region)

            image = image.resize((width, height), Image.BILINEAR)
            gt = gt.resize((width, height), Image.NEAREST)
            mask = mask.resize((width, height), Image.NEAREST)
            edge = edge.resize((width, height), Image.NEAREST)
            edge_distance = edge_distance.resize((width, height), Image.BILINEAR)
            pl = pl.resize((width, height), Image.NEAREST)
            prob = prob.resize((width, height), Image.BILINEAR)
            grey = grey.resize((width, height), Image.BILINEAR)

            (
                sample["image"],
                sample["gt"],
                sample["mask"],
                sample["edge"],
                sample["edge_distance"],
                sample["pl"],
                sample["prob"],
                sample["grey"],
            ) = (image, gt, mask, edge, edge_distance, pl, prob, grey)

            samples[i] = sample

        return samples
