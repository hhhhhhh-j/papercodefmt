import numpy as np
import matplotlib.pyplot as plt

height = 256
width = 256

map_ = np.zeros((height, width), dtype=np.uint8)

    # 四周边框
map_[0, :] = 1
map_[-1, :] = 1
map_[:, 0] = 1
map_[:, -1] = 1

    # 中间横墙
wall_y = height // 2
gap = width // 4          # 每个缺口宽度 = 64

left = gap                # 64
right = width - gap       # 256 - 64 = 192

    # 横墙主体（画出 8 像素高的墙）
map_[wall_y - 4 : wall_y + 4, left:right] = 1



plt.imshow(map_, cmap='gray_r', origin='lower')
plt.show()

