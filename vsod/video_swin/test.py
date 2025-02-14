import torch
import torch.nn as nn
from torch.utils import data
from torchvision.transforms import functional as TF
import cv2
import argparse
from dataset.transforms import get_transforms
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dataset.data import VideoDataset
#from model.model_4frames_memory_st4 import model
from model.model_4frames_longterm_memory import model
import time
import numpy as np

model_name = "finetune_withSam_withRcf_30"

get_edge = False

t = 128
t1 = 120
t2 = 136

parser = argparse.ArgumentParser()


parser.add_argument("--size", default=256, type=int, help="image size")

args = parser.parse_args()


data_transforms = get_transforms(input_size=(args.size, args.size))
test_list = [
    "DAVIS",
    "DAVSOD",
    "FBMS",
    "SegTrack-V2",
    "ViSal",
    "VOS"
]

dataset = VideoDataset(
    root_dir="/root/dataset/WVtestset",
    trainingset_list=test_list,
    trainging=False,
    transforms=data_transforms,
    video_time_clip=4,
)


model = model(
    pretrained="./backbone/swin_base_patch4_window12_384_22k.pth", pretrained2d=True
)

# model = model(
#     pretrained="/root/guogy/video_swin/backbone/new_weight.pth",
#     dim_c=3,
#     dim_h=args.size,
#     dim_w=args.size,
# )


model.load_state_dict(torch.load("save_models/finetune/" + model_name + ".pth"))

model = model.cuda()

frames = 0
total_time = 0
print("Begin inference on {} {}.")
memory_tensor = None
flag = True
for data in dataset:
    preds = []

    testset_name = data[0]["name"].split("/")[-4]
    sequence_name = data[0]["name"].split("/")[-3]
    save_dir = os.path.join("results", model_name, testset_name, sequence_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.cuda.synchronize()
    start_time = time.time()

    frame_list = []
    for pack in data:
        images = pack["image"]
        images = images.unsqueeze(0)

        frame_list.append(images)

    input_frames = np.stack(frame_list, axis=2)  # b*3*t*h*w
    sal_fuse_inits, sal_refs, edge_maps,_ = model(torch.tensor(input_frames).cuda(), flag)
    # sal_fuse_inits, sal_refs, edge_maps, memory_tensor, flag = model(
    #         torch.tensor(input_frames).cuda(), memory_tensor, flag
    #     )
    preds = sal_refs

    torch.cuda.synchronize()
    end_time = time.time()

    a = end_time - start_time

    total_time = total_time + a
    frames = frames + 4

    if frames == 400:
        print("avg time", total_time / 400)

    for i in range(len(preds)):
        pred = preds[i]

        pred = 255 * torch.sigmoid(pred).data.cpu().squeeze().numpy()
        # pred[pred <= t] = 0
        # pred[pred > t] = 255
        #pred = 255 - pred
        

        image_name = data[i]["name"].split("/")[-1][:-3] + "png"
        image_forsave = cv2.resize(
            pred, (data[i]["original_width"], data[i]["original_height"])
        )
        cv2.imwrite(os.path.join(save_dir, image_name), image_forsave)
        print(os.path.join(save_dir, image_name))
