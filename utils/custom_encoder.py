import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        global_map_shape = observation_space["global_map"].shape   # (C,H,W)
        local_map_shape = observation_space["local_map"].shape   # (C,H,W)
        pose_dim  = observation_space["pose"].shape[0]

        self.cnn_global = nn.Sequential(
            nn.Conv2d(global_map_shape[0], 16, 5, 2, 2, bias=False), 
            nn.GroupNorm(8, 16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1, bias=False), 
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), 
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        with torch.no_grad():
            global_cnn_dim = self.cnn_global(torch.zeros(1, *global_map_shape)).shape[1]

        self.cnn_local = nn.Sequential(
            nn.Conv2d(local_map_shape[0], 16, 5, 2, 2, bias=False), 
            nn.GroupNorm(8, 16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1, bias=False), 
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), 
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        with torch.no_grad():
            local_cnn_dim = self.cnn_local(torch.zeros(1, *local_map_shape)).shape[1]

        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        self.fuse = nn.Sequential(
            nn.Linear(global_cnn_dim + local_cnn_dim + 64, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        g = observations["global_map"].float().contiguous()
        l = observations["local_map"].float().contiguous()
        p = observations["pose"].float().contiguous()
        feat_map_global  = self.cnn_global(g)
        feat_map_local  = self.cnn_local(l)
        feat_pose = self.pose_mlp(p)
        return self.fuse(torch.cat([feat_map_global, feat_map_local, feat_pose], dim=1))
