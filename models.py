import torch
import os
from torch import nn

# 定义一些基本的模块
class Transpose(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        """
        (B, S, C) or (B, C, S)
        """
        return x.transpose(1, 2)


class MlpBlock(nn.Module):
    """
    linear works on the last dim
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(), # 错了 应该是nn.GELU()
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.mlp(x)


class MixerBlock(nn.Module):
    def __init__(self, token_dim, channel_dim):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm([token_dim, channel_dim]),
            Transpose(),
            MlpBlock(token_dim),
            Transpose(),
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm([token_dim, channel_dim]),
            MlpBlock(channel_dim),
        )

    def forward(self, x):
        x = self.token_mixing(x) + x
        return x + self.channel_mixing(x)


class StratifiedPatch(nn.Module):
    def __init__(self, channel_dim, fig_size):
        super().__init__()
        self.channel_dim = channel_dim
        self.width, self.height = fig_size
        # slice the image three times, these kernel sizes are fixed here. You can also adapt them
        self.p_32 = nn.Conv2d(3, channel_dim, kernel_size=(32, 32), stride=(32, 32))
        self.p_vertical = nn.Conv2d(3, channel_dim, kernel_size=(16, 128), stride=(16, 128))
        self.p_horizontal = nn.Conv2d(3, channel_dim, kernel_size=(128, 16), stride=(128, 16))

    def forward(self, x):
        # x1 = self.p_16(x).view(x.size(0), self.channel_dim, -1)
        x1 = self.p_32(x).view(x.size(0), self.channel_dim, -1)
        # x3 = self.p_64(x).view(x.size(0), self.channel_dim, -1)
        x2 = self.p_vertical(x).view(x.size(0), self.channel_dim, -1)
        x3 = self.p_horizontal(x).view(x.size(0), self.channel_dim, -1)
        return torch.cat([x1, x2, x3], dim=2)

    def __get_token_dim(self):
        t_dim = sum([(self.width // p_s) * (self.height // p_s) for p_s in [32]])
        t_dim += sum([(self.width // p_x) * (self.height // p_y) for (p_x, p_y) in [(16, 128), (128, 16)]])
        return t_dim

    token_dim = property(__get_token_dim)


# MLP-Mixer model
class MLPMixer(nn.Module):
    def __init__(self, patch_size: int, channel_dim: int, num_blocks: int, fig_size):
        super().__init__()
        self.patch_size = patch_size
        # self.token_dim = sum([(width // p_s) * (height // p_s) for p_s in [16, 32, 64]])
        self.channel_dim = channel_dim
        self.num_blocks = num_blocks
        if self.patch_size < 0:
            self.patch_proj = StratifiedPatch(self.channel_dim, fig_size)
            self.token_dim = self.patch_proj.token_dim
        else:
            self.patch_proj = nn.Conv2d(3, channel_dim, kernel_size=(patch_size, patch_size),
                                        stride=(patch_size, patch_size))
            width, height = fig_size
            self.token_dim = (width // patch_size) * (height // patch_size)

        layers = [MixerBlock(self.token_dim, self.channel_dim) for _ in range(num_blocks)]
        self.mixer_mlp_blocks = nn.Sequential(*layers)
        self.out_LayerNorm = nn.LayerNorm([self.token_dim, self.channel_dim])
        self.out_fc = nn.Linear(self.channel_dim, 2)  # 可以直接在这里把2改成num_classes作为参数输入

    def forward(self, x):
        x = self.patch_proj(x).view(x.size(0), self.channel_dim, -1).transpose(1, 2)
        x = self.mixer_mlp_blocks(x)
        x = self.out_LayerNorm(x)
        x = self.out_fc(x.mean(axis=1))  # global avg. pooling

        return x


# if __name__ == '__main__':
#     model_ = MLPMixer(patch_size=30, channel_dim=20, num_blocks=5, fig_size=(267, 275))
#     print(model_.token_dim)
