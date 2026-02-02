import os
import math
import argparse
import numpy as np
import yaml
import rasterio
import imageio.v2 as imageio
from rasterio.fill import fillnodata
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent

class Args:
    tif = ROOT / "Dataset" / "dem" / "output.tin(2).tif"   # 输入 GeoTIFF DEM（高程栅格）
    out = ROOT / "Dataset" / "output" / "map"           # 输出目录：保存 tiles/ 和索引文件
    tile = 256                                          # 输出 tile 的大小：tile x tile（像素）
    overlap = 0.2                                       # 重叠率：0=无重叠；0.5=50%重叠；越大样本越多
    k = 20                                              # 代表性聚类簇数：最终大约选 k 张代表 tile
    seed = 0                                            # 随机种子：保证每次结果一致
    id = 20001                                          # 保存文件的id(起始记录编号)

def main():
    args = Args()

    os.makedirs(args.out, exist_ok=True) 
    tile_dir = os.path.join(args.out, "tiles")
    os.makedirs(tile_dir, exist_ok=True)
    png_dir = args.out / "tiles_png"
    png_dir.mkdir(parents=True, exist_ok=True)

    stride = int(args.tile * (1.0 - args.overlap))
    stride = max(1, stride)

    with rasterio.open(args.tif) as src:
        dem = src.read(1)
        nodata = src.nodata
        transform = src.transform
        crs = str(src.crs)
        res = src.res

    if nodata is None:
        raise ValueError("GeoTIFF 没有 nodata 标记，建议先补上 nodata 或自行做 mask。")

    dem_filled = fill_nodata_raster(dem, nodata_value=float(nodata))
    dem_padded, pad_h, pad_w = pad_to_tiles(dem_filled, args.tile, stride, pad_mode="edge")

    hp, wp = dem_padded.shape
    tiles_meta = []
    feats = []

    tile_id = args.id
    for row in range(0, hp - args.tile + 1, stride):
        for col in range(0, wp - args.tile + 1, stride):
            tile = dem_padded[row:row + args.tile, col:col + args.tile].astype(np.float32)

            # 保存 tile
            np.save(os.path.join(tile_dir, f"tile_{tile_id:05d}.npy"), tile)

            # --- 保存灰度 PNG（仅用于可视化）---
            t = tile.astype(np.float32)

            # 用 2%~98% 分位裁剪，避免极端值导致整张图发黑/发白（更好看）
            lo = np.percentile(t, 2)
            hi = np.percentile(t, 98)
            t_clip = np.clip(t, lo, hi)

            # 归一化到 0~255
            img_u8 = (255 - ((t_clip - lo) / (hi - lo + 1e-6) * 255.0)).astype(np.uint8)

            # 写文件
            imageio.imwrite(png_dir / f"tile_{tile_id:05d}.png", img_u8)

            # tile 左上角的空间坐标
            x0, y0 = transform * (col, row)

            f = compute_features(tile)
            feats.append([f["mean"], f["std"], f["range"], f["grad_mean"], f["grad_std"], f["lap_var"]])

            tiles_meta.append({
                "tile_id": tile_id,
                "row": int(row),
                "col": int(col),
                "origin_x": float(x0),
                "origin_y": float(y0),
                "features": f
            })
            tile_id += 1

    feats = np.asarray(feats, dtype=np.float32)
    n_tiles = feats.shape[0]
    k = min(args.k, n_tiles)

    # 代表性选择：KMeans + 每类选离质心最近的 tile
    representative = []
    if n_tiles > 0:
        km = KMeans(n_clusters=k, random_state=args.seed, n_init="auto")
        labels = km.fit_predict(feats)
        centers = km.cluster_centers_
        for c in range(k):
            idx = np.where(labels == c)[0]
            if idx.size == 0:
                continue
            d = np.linalg.norm(feats[idx] - centers[c], axis=1)
            representative.append(int(idx[np.argmin(d)]))

    # 写 index
    index = {
        "source": os.path.basename(args.tif),
        "crs": crs,
        "resolution": {"x": float(res[0]), "y": float(res[1])},
        "nodata_value_in_source": float(nodata),
        "tile_size": int(args.tile),
        "stride": int(stride),
        "overlap": float(args.overlap),
        "padding": {"pad_h": int(pad_h), "pad_w": int(pad_w), "mode": "edge"},
        "tiles": tiles_meta,
        "representative_tile_ids": representative
    }

    with open(os.path.join(args.out, "tiles_index.yaml"), "w") as f:
        yaml.safe_dump(index, f, sort_keys=False, allow_unicode=True)

    with open(os.path.join(args.out, "representative_ids.txt"), "w") as f:
        for rid in representative:
            f.write(f"{rid}\n")

    print(f"[OK] tiles: {n_tiles}, representative: {len(representative)}")
    print(f"out: {args.out}")

def compute_features(tile: np.ndarray) -> dict:
    t = tile.astype(np.float32)
    gy, gx = np.gradient(t)
    gmag = np.sqrt(gx * gx + gy * gy)
    lap = ndi.laplace(t)
    return {
        "mean": float(np.mean(t)),
        "std": float(np.std(t)),
        "min": float(np.min(t)),
        "max": float(np.max(t)),
        "range": float(np.max(t) - np.min(t)),
        "grad_mean": float(np.mean(gmag)),
        "grad_std": float(np.std(gmag)),
        "lap_var": float(np.var(lap)),
    }

def fill_nodata_raster(arr: np.ndarray, nodata_value: float,
                      max_search_distance: int = 100) -> np.ndarray:
    arr = arr.astype(np.float32)
    nan_mask = (arr == nodata_value)
    if not np.any(nan_mask):
        return arr

    # GDALFillNodata 要求 mask=1 表示有效像元
    img = np.where(nan_mask, 0.0, arr).astype(np.float32)
    valid = (~nan_mask).astype(np.uint8)
    filled = fillnodata(
        img,
        mask=valid,
        max_search_distance=max_search_distance,
        smoothing_iterations=0
    )
    return filled.astype(np.float32)

def pad_to_tiles(arr: np.ndarray, tile: int, stride: int, pad_mode: str = "edge"):
    h, w = arr.shape
    # 至少补到 tile 大小
    pad_h = max(0, tile - h)
    pad_w = max(0, tile - w)

    # 若比 tile 大，为了完整滑窗覆盖，再补到 stride 对齐
    if h > tile:
        pad_h = (stride - (h - tile) % stride) % stride
    if w > tile:
        pad_w = (stride - (w - tile) % stride) % stride

    arr_p = np.pad(arr, ((0, pad_h), (0, pad_w)), mode=pad_mode)
    return arr_p, pad_h, pad_w

if __name__ == "__main__":
    main()
