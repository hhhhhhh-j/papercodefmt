import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent

def read_dem():
    '''
    return self.global_map
    '''
    file_path = ROOT / "Dataset" / "output" / "map" / "tiles" / "tile_20012.npy"
    tile = np.load(file_path).astype(np.float32)

    lo = np.percentile(tile, 2)
    hi = np.percentile(tile, 98)
    tile_clip = np.clip(tile, lo, hi)

    m01 = (tile_clip - lo) / (hi - lo + 1e-6)   # 0~1 float32

    # log
    logger.debug("m01:{}", m01)
    logger.info(f"tile shape={tile.shape} dtype={tile.dtype} min/max={tile.min():.3f}/{tile.max():.3f}")
    logger.info(f"m01  shape={m01.shape} dtype={m01.dtype} min/max={m01.min():.3f}/{m01.max():.3f}")
    logger.info(f"m01 mean/std={m01.mean():.3f}/{m01.std():.3f}")
    logger.info(f"m01 p1/p50/p99={np.percentile(m01,[1,50,99])}")

    return m01

if __name__ == "__main__":
    map = read_dem()
    plt.figure()
    plt.imshow(map, cmap="gray_r", vmin=0, vmax=1)
    plt.colorbar()
    plt.title("m01 (0~1)")
    plt.show()