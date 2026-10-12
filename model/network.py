# Copyright 2026 The HTD Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from functools import partial


class BrainNetwork_mindeye_2d_light(nn.Module):
    def __init__(self, out_dim=768, in_dim=30976, clip_size=768, h=4096, n_blocks=4, act_first=False, n_times=5, normalization="bn", activation="relu", projection="nonlinear"):
        super().__init__()
        self.n_times=n_times
        norm_func = partial(nn.BatchNorm1d, num_features=1) if normalization=="bn" else partial(nn.LayerNorm, normalized_shape=h)
        norm_func2 = partial(nn.BatchNorm1d, num_features=1) if normalization=="bn" else partial(nn.LayerNorm, normalized_shape=2048)
        norm_func3 = partial(nn.BatchNorm1d, num_features=1) if normalization=="bn" else partial(nn.LayerNorm, normalized_shape=clip_size)
        act_fn = nn.ReLU if activation == "relu" else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)

        norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(self.n_times, 32, kernel_size=9, stride=3, padding=4, bias=False)
        self.bn1 = norm_layer(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = norm_layer(48)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = norm_layer(64)
        self.relu = nn.ReLU()

        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.5),
        )
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])
        self.lin1=nn.Linear(h, out_dim)
        self.n_blocks = n_blocks
        self.clip_size = clip_size
        self.projection = projection

        if self.projection=="linear":
            self.projector = nn.Linear(clip_size, clip_size)
        elif self.projection=="none":
            self.projector = nn.Identity()
        elif self.projection=="nonlinear":
            self.projector = nn.Sequential(
                norm_func3(),
                act_fn(),
                nn.Linear(clip_size, 2048),
                norm_func2(),
                act_fn(),
                nn.Linear(2048, 2048),
                norm_func2(),
                act_fn(),
                nn.Linear(2048, clip_size)
            )
        
    def encode_fmri(self, x, debug=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = x.reshape(len(x),-1)

        x = self.lin0(x)  # bs, t, h
        for res_block in range(self.n_blocks):
            residual = x
            x = self.mlp[res_block](x)
            x = x + residual
        x = x.reshape(len(x), -1)
        h_t = x
        x_pre = x
        x = self.lin1(x)
        fmri_feature = self.projector(x.reshape(len(x), -1, self.clip_size))
        if debug:
            return {"x_pre":x_pre,"x_fused":x,"h_t":h_t,"fmri_feature":fmri_feature}
        return x, fmri_feature
    
class BrainNetwork_mindeye_2d_fuse(nn.Module):
    def __init__(self, out_dim=768, in_dim=30976, clip_size=768, h=4096, n_blocks=4, act_first=False, n_times=5, fuse_method="mlp", normalization="bn", activation="relu", projection="nonlinear"):
        super().__init__()
        self.n_times=n_times
        norm_func = partial(nn.BatchNorm1d, num_features=1) if normalization=="bn" else partial(nn.LayerNorm, normalized_shape=h)
        norm_func2 = partial(nn.BatchNorm1d, num_features=1) if normalization=="bn" else partial(nn.LayerNorm, normalized_shape=2048)
        norm_func3 = partial(nn.BatchNorm1d, num_features=1) if normalization=="bn" else partial(nn.LayerNorm, normalized_shape=clip_size)
        act_fn = nn.ReLU if activation == "relu" else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)

        norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(1, 32, kernel_size=9, stride=3, padding=4, bias=False)
        self.bn1 = norm_layer(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = norm_layer(48)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = norm_layer(64)
        self.relu = nn.ReLU()

        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h),
            *[item() for item in act_and_norm],
            nn.Dropout(0.5),
        )
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])
        self.fuse_method=fuse_method
        if self.fuse_method=="mlp":
            self.fuse = nn.Linear(h*self.n_times, h)
        elif self.fuse_method=="rnn_last" or self.fuse_method=="rnn_average":
            self.fuse = nn.RNN(h,hidden_size=h,batch_first=True)
        elif self.fuse_method=="average":
            self.fuse = nn.Identity()
        else:
            raise Exception
        self.lin1=nn.Linear(h, out_dim)
        self.n_blocks = n_blocks
        self.clip_size = clip_size
        self.projection = projection

        if self.projection=="linear":
            self.projector = nn.Linear(clip_size, clip_size)
        elif self.projection=="none":
            self.projector = nn.Identity()
        elif self.projection=="nonlinear":
            self.projector = nn.Sequential(
                norm_func3(),
                act_fn(),
                nn.Linear(clip_size, 2048),
                norm_func2(),
                act_fn(),
                nn.Linear(2048, 2048),
                norm_func2(),
                act_fn(),
                nn.Linear(2048, clip_size)
            )
        
    def encode_fmri(self, x, debug=False):
        B,T,H,W=x.shape
        x = x.view(B*T,1,H,W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = x.reshape(B*T,1,-1)

        x = self.lin0(x)  # bs, t, h
        for res_block in range(self.n_blocks):
            residual = x
            x = self.mlp[res_block](x)
            x = x + residual
        x = x.reshape(B,T,-1)

        h_t = x
        if self.fuse_method=="mlp":
            x = x.reshape(len(x), -1)
            x = self.fuse(x)
        elif self.fuse_method=="rnn_average":
            x,_ = self.fuse(x)
            x = x.mean(dim=1)
        elif self.fuse_method=="rnn_last":
            x,_ = self.fuse(x)
            x = x[:,-1]
        elif self.fuse_method=="average":
            x = self.fuse(x).mean(dim=1)
        x_pre = x
        x = self.lin1(x)
        fmri_feature = self.projector(x.reshape(len(x), -1, self.clip_size))
        if debug:
            return {"x_pre":x_pre,"x_fused":x,"h_t":h_t,"fmri_feature":fmri_feature}
        return x, fmri_feature