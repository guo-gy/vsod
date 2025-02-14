from torch.utils.data import Dataset
import os
import random
from PIL import Image
import numpy as np


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir="/root/guogy/dataset/",
        trainingset_list=["DAVSOD", "DAVIS"],
        video_time_clip=4,
        time_interval=1,
        trainging=True,
        transforms=None,
    ):
        super(VideoDataset, self).__init__()
        self.root_dir = root_dir
        self.transforms = transforms

        self.time_clips = video_time_clip  # must be <=4
        self.clips = []

        self.training = trainging

        sequence_list = []
        # shuffle
        for trainset in trainingset_list:
            video_root = os.path.join(root_dir, trainset)
            sequences = sorted(os.listdir(video_root))
            for sequence in sequences:
                sequence_list.append((trainset, sequence))

        random.shuffle(sequence_list)
        for trainset, sequence in sequence_list:
            sequence_info = self.get_frame_list(trainset, sequence)
            # print(len(sequence_info))
            self.clips += self.get_clips(sequence_info, sequence)

    def get_frame_list(self, trainset, sequence):
        image_path_root = os.path.join(self.root_dir, trainset, sequence, "Imgs")
        frame_list = sorted(os.listdir(image_path_root))
        sequence_info = []
        for i in range(len(frame_list)):
            frame_info = {
                "image_path": os.path.join(
                    self.root_dir, trainset, sequence, "Imgs", frame_list[i]
                ),
                "mask_path": os.path.join(
                    self.root_dir, trainset, sequence, "mask", frame_list[i]
                ),
                "gt_path": os.path.join(
                    self.root_dir, trainset, sequence, "gt", frame_list[i]
                ),
                "edge_path": os.path.join(
                    self.root_dir, trainset, sequence, "edge", frame_list[i]
                ),
                "edge_distance_path": os.path.join(
                    self.root_dir, trainset, sequence, "edge_distance", frame_list[i]
                ),
                "pl_path": os.path.join(
                    self.root_dir, trainset, sequence, "PL", frame_list[i]
                ),
                "prob_path": os.path.join(
                    self.root_dir, trainset, sequence, "Prob", frame_list[i]
                ),
                "grey_path": os.path.join(
                    self.root_dir, trainset, sequence, "grey", frame_list[i]
                ),
            }
            sequence_info.append(frame_info)

        return sequence_info

    def get_clips(self, sequence_info, sequence_name):
        clips = []
        num_clips = int(len(sequence_info) / self.time_clips)

        for i in range(num_clips):
            is_first_clip = i == 0  # 第一个4帧组标记为True
            # print(f'is_first_clip is {is_first_clip}')
            clips.append(
                {
                    "frames": sequence_info[
                        self.time_clips * i : self.time_clips * (i + 1)
                    ],
                    "is_first_clip": is_first_clip,
                    "sequence_name": sequence_name,
                }
            )

        # 如果有剩余的帧，创建最后一个剪辑
        finish = self.time_clips * num_clips
        if finish < len(sequence_info):
            clips.append(
                {
                    "frames": sequence_info[
                        len(sequence_info) - self.time_clips : len(sequence_info)
                    ],
                    "is_first_clip": False,
                    "sequence_name": sequence_name,
                }
            )

        return clips

    def get_frame(self, frame_info):
        image_path = frame_info["image_path"]

        image = Image.open(image_path).convert("RGB")
        image_size = image.size[:2]

        if self.training:
            mask_path = frame_info["mask_path"]
            gt_path = frame_info["gt_path"]
            edge_path = frame_info["edge_path"]
            edge_distance_path = frame_info["edge_distance_path"]
            pl_path = frame_info["pl_path"]
            prob_path = frame_info["prob_path"]
            grey_path = frame_info["grey_path"]

            mask = Image.open(mask_path).convert("L")
            gt = Image.open(gt_path).convert("L")
            edge = Image.open(edge_path).convert("L")
            edge_distance = Image.open(edge_distance_path).convert("L")
            pl = Image.open(pl_path).convert("L")
            prob = Image.open(prob_path).convert("L")
            grey = Image.open(grey_path).convert("L")

        else:
            mask = None
            gt = None
            edge = None
            edge_distance = None
            pl = None
            prob = None
            grey = None

        sample = {
            "image": image,
            "mask": mask,
            "gt": gt,
            "edge": edge,
            "edge_distance": edge_distance,
            "pl": pl,
            "prob": prob,
            "grey": grey,
        }

        sample["name"] = image_path
        sample["original_height"] = image_size[1]
        sample["original_width"] = image_size[0]

        return sample

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        clip_frames = clip_info["frames"]
        is_first_clip = clip_info["is_first_clip"]
        # print(f'is_first_clip is {is_first_clip}')
        sequence_name = clip_info["sequence_name"]

        clip_output = []
        # # Random reverse when training
        # if self.training and random.randint(0, 1):
        #     clip_frames = clip_frames[::-1]

        for i in range(len(clip_frames)):
            item = self.get_frame(clip_frames[i])
            clip_output.append(item)

        clip_output = self.transforms(clip_output)

        return {
            "clip": clip_output,
            "is_first_clip": is_first_clip,
            "sequence_name": sequence_name,
        }

    def __len__(self):
        return len(self.clips)
