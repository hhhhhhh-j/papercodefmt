import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        map_shape = observation_space["map"].shape  # (2, 64, 64)
        channels = map_shape[0]

        # CNN 提取器
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Flatten()
        )

        # 计算 CNN 输出维度：自动计算，不用你算
        with torch.no_grad():
            sample = torch.zeros(1, *map_shape)
            cnn_output_dim = self.cnn(sample).shape[1]

        # 再用 MLP 压缩到 features_dim
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        map_tensor = observations["map"]
        cnn_features = self.cnn(map_tensor)
        return self.linear(cnn_features)
