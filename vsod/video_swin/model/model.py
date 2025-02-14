import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.edge import Edge_Module, RCAB
from utils.label import Label
from video_swin.backbone.swin_transformer import SwinTransformer3D


class ConvBnRelu(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride, kernel_size, padding, relu=True
    ):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if relu == True:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        if self.relu is not None:
            x = self.relu(x)

        return x


def swin_base(pretrained, pretrained2d):
    # initialize the SwinTransformer backbone
    backbone = SwinTransformer3D(
        patch_size=(1, 4, 4),
        depths=[2, 2, 18, 2],
        embed_dim=128,
        num_heads=[4, 8, 16, 32],
        window_size=(8, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        pretrained2d=pretrained2d,
    )
    if pretrained:
        print("Initializing Swin Transformer weights")
        backbone.init_weights(pretrained=pretrained)
    else:
        print("Randomly initialize Swin Transformer weights.")
        backbone.init_weights()
    return backbone


def update_mem(mem_pool, tmp, flag):
    tmp = tmp.unsqueeze(2)
    if flag:
        frames = tmp.detach()
        return frames
    _, _, t, _, _ = mem_pool.shape
    if t < 4:
        frames = torch.cat(
            (mem_pool, tmp.detach()),
            dim=2,
        )
        return frames
    frames = torch.cat(
        (mem_pool[:, :, 1:, :, :], tmp.detach()),
        dim=2,
    )
    return frames


class CrossAttention(nn.Module):
    def __init__(self, c, hidden_dim=512):
        super(CrossAttention, self).__init__()

        # 线性层将输入的特征映射到查询、键和值的维度
        self.query_video = nn.Conv3d(c, hidden_dim, kernel_size=1)
        self.key_memory = nn.Conv3d(c, hidden_dim, kernel_size=1)
        self.value_video = nn.Conv3d(c, hidden_dim, kernel_size=1)

        self.hidden_dim = hidden_dim

    def forward(self, video_features, memory_features):
        b, c, h, w = video_features.shape
        t = 1
        video_features = video_features.reshape(b, c, t, h, w)
        _, _, n, _, _ = memory_features.shape

        # 将视频特征和记忆特征映射为查询、键和值
        query = self.query_video(video_features).view(
            b, self.hidden_dim, 1 * h * w
        )  # (b, hidden_dim, t*h*w)
        key = self.key_memory(memory_features).view(
            b, self.hidden_dim, n * h * w
        )  # (b, hidden_dim, n*h*w)
        value = self.value_video(memory_features).view(
            b, self.hidden_dim, n * h * w
        )  # (b, hidden_dim, n*h*w)

        # 计算注意力得分 (b, hidden_dim, t*h*w) 和 (b, hidden_dim, n*h*w)
        attention_scores = torch.bmm(query.permute(0, 2, 1), key)  # (b, t*h*w, n*h*w)
        attention_scores = attention_scores / (self.hidden_dim**0.5)  # 缩放
        attention_weights = F.softmax(attention_scores, dim=-1)  # (b, t*h*w, n*h*w)

        # 计算加权值 (b, t*h*w, hidden_dim)
        attention_output = torch.bmm(
            attention_weights, value.permute(0, 2, 1)
        )  # (b, t*h*w, hidden_dim)
        attention_output = attention_output.permute(0, 2, 1).reshape(
            b, self.hidden_dim, h, w
        ) 

        return attention_output


class SelfAttention(nn.Module):
    def __init__(self, c, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv3d(c, hidden_dim, kernel_size=1)
        self.key = nn.Conv3d(c, hidden_dim, kernel_size=1)
        self.value = nn.Conv3d(c, hidden_dim, kernel_size=1)
        self.hidden_dim = hidden_dim

    def forward(self, video_features):
        b, c, t, h, w = video_features.shape
        query = self.query(video_features).view(b, self.hidden_dim, t * h * w)
        key = self.key(video_features).view(b, self.hidden_dim, t * h * w)
        value = self.value(video_features).view(b, self.hidden_dim, t * h * w)
        attention_scores = torch.bmm(query.permute(0, 2, 1), key)
        attention_scores = attention_scores / (self.hidden_dim**0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.bmm(attention_weights, value.permute(0, 2, 1))
        attention_output = (
            attention_output.permute(0, 2, 1)
            .reshape(b, self.hidden_dim, t, h, w)
            .sum(dim=2)
            .reshape(b, self.hidden_dim, 1, h, w)
        )
        return attention_output


class model(nn.Module):
    def __init__(self, pretrained, pretrained2d):
        super(model, self).__init__()
        self.video_encoder = swin_base(pretrained=pretrained, pretrained2d=pretrained2d)
        self.video_decoder = VideoDecoder()
        self.label_layer = Label(in_fea=[128, 256, 512, 1024], mid_fea=32)
        # self.num_layers = 16

        self.crossAttention3 = CrossAttention(1024, 1024)
        self.crossAttention2 = CrossAttention(512, 512)
        self.crossAttention1 = CrossAttention(256, 256)
        self.crossAttention0 = CrossAttention(128, 128)
        self.selfAttention3 = SelfAttention(1024, 1024)
        self.selfAttention2 = SelfAttention(512, 512)
        self.selfAttention1 = SelfAttention(256, 256)
        self.selfAttention0 = SelfAttention(128, 128)
        self.labelselfAttention3 = SelfAttention(1024, 1024)
        self.labelselfAttention2 = SelfAttention(512, 512)
        self.labelselfAttention1 = SelfAttention(256, 256)
        self.labelselfAttention0 = SelfAttention(128, 128)

        self.memory_slot3 = torch.empty(1, 1024, device="cuda")
        self.memory_slot2 = torch.empty(1, 512, device="cuda")
        self.memory_slot1 = torch.empty(1, 256, device="cuda")
        self.memory_slot0 = torch.empty(1, 128, device="cuda")

    def forward(self, input_frames, flag):
        video_features = self.video_encoder(input_frames)
        
        b, c, t, h, w = video_features[0].shape
        stage = 4

        # 如果是第一帧，清空
        if flag:
            self.memory_slot3 = torch.empty(1, 1024, device="cuda")
            self.memory_slot2 = torch.empty(1, 512, device="cuda")
            self.memory_slot1 = torch.empty(1, 256, device="cuda")
            self.memory_slot0 = torch.empty(1, 128, device="cuda")

        sal_fuse_inits = []
        sal_refs = []
        edge_maps = []
        labels = []
        
        # 遍历这四帧
        for i in range(t):
            # 获取帧特征
            tmp = []
            frame_features = []
            for j in range(stage):
                cur_feature = video_features[j][:, :, i, :, :]
                frame_features.append(cur_feature)
                tmp.append(cur_feature)
            if not flag:
                self.longterm_memory3 = self.selfAttention3(self.memory_slot3)
                self.longterm_memory2 = self.selfAttention2(self.memory_slot2)
                self.longterm_memory1 = self.selfAttention1(self.memory_slot1)
                self.longterm_memory0 = self.selfAttention0(self.memory_slot0)
                self.labellongterm_memory3 = self.labelselfAttention3(self.memory_slot3)
                self.labellongterm_memory2 = self.labelselfAttention2(self.memory_slot2)
                self.labellongterm_memory1 = self.labelselfAttention1(self.memory_slot1)
                self.labellongterm_memory0 = self.labelselfAttention0(self.memory_slot0)
                # 融合记忆
                frame_features[3] = self.crossAttention3(
                    frame_features[3], self.longterm_memory3
                )
                frame_features[2] = self.crossAttention2(
                    frame_features[2], self.longterm_memory2
                )
                frame_features[1] = self.crossAttention1(
                    frame_features[1], self.longterm_memory1
                )
                frame_features[0] = self.crossAttention0(
                    frame_features[0], self.longterm_memory0
                )
            # 更新记忆
            self.memory_slot3 = update_mem(self.memory_slot3.detach(), tmp[3], flag)
            self.memory_slot2 = update_mem(self.memory_slot2.detach(), tmp[2], flag)
            self.memory_slot1 = update_mem(self.memory_slot1.detach(), tmp[1], flag)
            self.memory_slot0 = update_mem(self.memory_slot0.detach(), tmp[0], flag)
            # 解码
            label = None
            if not flag:
                label = self.label_layer(
                    self.labellongterm_memory0.squeeze(2),
                    self.labellongterm_memory1.squeeze(2),
                    self.labellongterm_memory2.squeeze(2),
                    self.labellongterm_memory3.squeeze(2),
                )
            sal_fuse_init, sal_ref, edge_map = self.video_decoder(frame_features)
            sal_fuse_inits.append(sal_fuse_init)
            sal_refs.append(sal_ref)
            edge_maps.append(edge_map)
            labels.append(label)
            flag = False
        return sal_fuse_inits, sal_refs, edge_maps, labels


class VideoDecoder(nn.Module):
    def __init__(self):
        super(VideoDecoder, self).__init__()

        self.conv43 = ConvBnRelu(
            in_channels=1024 + 512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            relu=True,
        )
        self.conv432 = ConvBnRelu(
            in_channels=512 + 256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            relu=True,
        )
        self.conv4321 = ConvBnRelu(
            in_channels=256 + 128,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            relu=True,
        )

        self.edge_layer = Edge_Module(in_fea=[128, 256, 512, 1024], mid_fea=32)
        self.pred = nn.Conv2d(32, 1, kernel_size=1, stride=1)

        self.sal_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.edge_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.rcab_sal_edge = RCAB(64)
        self.fused_edge_sal = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, frame_features):
        
        x1, x2, x3, x4 = frame_features  # b,c,h,w
        edge_map = self.edge_layer(x1, x2, x3, x4)

        conv4 = F.upsample(x4, scale_factor=2, mode="bilinear", align_corners=True)

        conv43 = self.conv43(torch.cat((conv4, x3), dim=1))
        conv43 = F.upsample(conv43, scale_factor=2, mode="bilinear", align_corners=True)

        conv432 = self.conv432(torch.cat((conv43, x2), dim=1))
        conv432 = F.upsample(
            conv432, scale_factor=2, mode="bilinear", align_corners=True
        )

        conv4321 = self.conv4321(torch.cat((conv432, x1), dim=1))
        fuse_feature = F.upsample(conv4321, scale_factor=4, mode="bilinear")

        sal_fuse_init = self.pred(fuse_feature)

        sal_feature = F.relu(self.sal_conv(sal_fuse_init))
        edge_feature = F.relu(self.edge_conv(edge_map))
        sal_edge_feature = torch.cat((sal_feature, edge_feature), dim=1)
        sal_edge_feature = self.rcab_sal_edge(sal_edge_feature)
        sal_ref = self.fused_edge_sal(sal_edge_feature)

        return sal_fuse_init, sal_ref, edge_map
