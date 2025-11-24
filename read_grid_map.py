import cv2
import numpy as np
import matplotlib.pyplot as plt

class ReadGridMap:
    def __init__(self):
        self.grid_map = None
        self.small_map = None

    def convert(self, map_path):
        # 读取PGM图像
        image = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot load image at {map_path}")

        # 将图像转换为二值化地图
        self.grid_map = 1 - image / 255  # 归一化到0-1
        self.small_map = cv2.resize(self.grid_map, (512, 512), interpolation=cv2.INTER_NEAREST)
        return self.grid_map,self.small_map
    
    def lookit(self):
        if self.grid_map is None:
            raise ValueError("Grid map is not initialized. Please run convert() first.")
        plt.imshow(self.grid_map, cmap='binary', origin='lower')
        plt.show()

if __name__ == "__main__":
    map_path = "/home/fmt/decision_making/sb3_SAC/map/20220630.pgm"
    converter = ReadGridMap()
    grid_map,small_map = converter.convert(map_path)
    print(grid_map)
    print('-----------------')
    print(small_map)
    plt.imshow(small_map, cmap='binary', origin='lower')
    plt.show()
