import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
from torch.utils import data
from datetime import datetime
import argparse
from video_swin.dataset.data import VideoDataset
from dataset.transforms import get_train_transforms
from torch.utils.data import DataLoader
from video_swin.model.model import model
import torch.optim as optim
from datetime import datetime
from utils import smoothness
from torch.nn import functional as F
import numpy as np

# 选择约束方案
use_SAM = True  # 使用SAM强化标签
use_RCF = True  # 使用RCF边缘图约束边缘
use_longS = True # 使用长期记忆辅助监督

parser = argparse.ArgumentParser()

parser.add_argument("--epoch", type=int, default=30, help="epoch number")
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
parser.add_argument("--batchsize", type=int, default=1, help="training batch size")
parser.add_argument("--size", type=int, default=256, help="training dataset size")
parser.add_argument("--clip", type=float, default=0.5, help="gradient clipping margin")
parser.add_argument(
    "--decay_rate", type=float, default=0.9, help="decay rate of learning rate"
)
parser.add_argument(
    "--decay_epoch", type=int, default=15, help="every n epochs decay learning rate"
)
args = parser.parse_args()

transforms = get_train_transforms(input_size=(args.size, args.size))
dataset = VideoDataset(
    root_dir="/root/guogy/dataset",
    trainingset_list=["DAVIS", "DAVSOD"],
    trainging=True,
    transforms=transforms,
)

train_dataloader = DataLoader(
    dataset=dataset,
    batch_size=args.batchsize,
    num_workers=6,
    shuffle=False,
    drop_last=False,
)

model = model(
    pretrained="./backbone/swin_base_patch4_window12_384_22k.pth", pretrained2d=True
)


model = model.cuda()
print("network ready!!!")

optimizer = optim.Adam(model.parameters(), lr=args.lr)

CE = torch.nn.BCELoss()
smooth_loss = smoothness.smoothness_loss(size_average=True)


def train(train_dataloader, model, optimizer, epoch, total_epoch):
    model.train()

    total_step = len(train_dataloader)

    memory_tensor = None
    # packs: a clip, 4 frames of a sequence
    for i, clip_info in enumerate(train_dataloader):
        optimizer.zero_grad()

        packs = clip_info["clip"]
        flag = clip_info["is_first_clip"]
        sequence_name = clip_info["sequence_name"]

        # if flag:
        # print(f"sequence_name: {sequence_name}, flag: {flag}")
        frames = []
        for pack in packs:
            image = pack["image"]
            frames.append(image)

        input_frames = np.stack(frames, axis=2)  # b*3*t*h*w
        input_frames = torch.tensor(input_frames)
        # print(input_frames.shape)

        sal_fuse_inits, sal_refs, edge_maps, labels = model(input_frames.cuda(), flag)

        loss = 0
        # print(len(sal_refs))
        for k in range(len(sal_fuse_inits)):
            sal1 = sal_fuse_inits[k]
            sal2 = sal_refs[k]
            edge_map = edge_maps[k]
            label = labels[k]
            edge = packs[k]["edge"].cuda()
            mask = packs[k]["mask"].cuda()
            pl = packs[k]["pl"].cuda()
            prob = packs[k]["prob"].cuda()
            gt = packs[k]["gt"].cuda()
            edge_distance = packs[k]["edge_distance"].cuda()
            grey = packs[k]["grey"].cuda()

            sal_prob1 = torch.sigmoid(sal1)
            sal_prob2 = torch.sigmoid(sal2)
            edge_prob = torch.sigmoid(edge_map)
            if label != None and use_longS:
                label_prob = torch.sigmoid(label)
            # sal_prob = STE(sal_prob).clamp(min=0.0, max=1.0)

            if use_SAM:
                sal_loss1 = F.binary_cross_entropy(sal_prob1, pl) + 0.3 * smooth_loss(
                    sal_prob1, grey
                )
                sal_loss2 = F.binary_cross_entropy(sal_prob2, pl) + 0.3 * smooth_loss(
                    sal_prob2, grey
                )
                loss += sal_loss1 + sal_loss2
                if label!=None and use_longS:
                    sal_loss3 = F.binary_cross_entropy(label_prob, pl)
                    sal_loss4 = F.binary_cross_entropy(sal_prob2, label_prob)
                    loss += sal_loss3 + sal_loss4
                # sal_loss2 = F.binary_cross_entropy(sal_prob, prob)
            else:
                sal_loss1 = F.binary_cross_entropy(
                    sal_prob1 * mask, gt
                ) + 0.3 * smooth_loss(sal_prob1, grey)
                sal_loss2 = F.binary_cross_entropy(
                    sal_prob2 * mask, gt
                ) + 0.3 * smooth_loss(sal_prob2, grey)
                # sal_loss2 = F.binary_cross_entropy(sal_prob * mask, gt)
                loss += sal_loss1 + sal_loss2

            if use_RCF:
                edge = (torch.abs(sal_prob1 - 0.5) < 0.45).type(torch.IntTensor).cuda()
                edge_loss1 = (edge_prob * edge_distance).mean()
                edge_loss2 = (edge * edge_distance).mean()
                loss += edge_loss1 + 2 * edge_loss2
        loss.backward()
        optimizer.step()

        if i % 100 == 0 or i == total_step:
            print(
                "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:0.4f}, sal loss1: {:0.4f},sal loss2: {:0.4f}".format(
                    datetime.now(),
                    epoch,
                    args.epoch,
                    i,
                    total_step,
                    loss.data,
                    sal_loss1.data,
                    sal_loss2.data,
                )
            )
            if use_RCF:
                print(
                    ", edge loss1: {:0.4f}, edge loss2: {:0.4f}".format(
                        edge_loss1.data, edge_loss2.data
                    )
                )

    save_path = "save_models/finetune/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_name = "finetune"

    if use_SAM:
        model_name += "_withSam"

    if use_RCF:
        model_name += "_withRcf"

    if epoch % 5 == 0 and epoch < (total_epoch - 5):
        torch.save(model.state_dict(), save_path + model_name + "_%d" % epoch + ".pth")

    if epoch >= (total_epoch - 5):
        torch.save(model.state_dict(), save_path + model_name + "_%d" % epoch + ".pth")


print("start training!!!")


def adjust_lr(optimizer, epoch, decay_rate=0.9, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    print("lr=", optimizer.param_groups[0]["lr"])

    for param_group in optimizer.param_groups:
        param_group["lr"] *= decay


for epoch in range(1, args.epoch + 1):
    adjust_lr(optimizer, epoch, args.decay_rate, args.decay_epoch)

    train(train_dataloader, model, optimizer, epoch, args.epoch)
