import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        map_shape = observation_space["map"].shape   # (C,H,W)
        pose_dim  = observation_space["pose"].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(map_shape[0], 16, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            cnn_dim = self.cnn(torch.zeros(1, *map_shape)).shape[1]

        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )

        self.fuse = nn.Sequential(
            nn.Linear(cnn_dim + 64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        feat_map  = self.cnn(observations["map"])
        feat_pose = self.pose_mlp(observations["pose"])
        return self.fuse(torch.cat([feat_map, feat_pose], dim=1))
