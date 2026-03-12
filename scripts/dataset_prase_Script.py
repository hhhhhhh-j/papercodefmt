import os
import math
import argparse
import cv2
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
    tif = ROOT / "Dataset" / "dem" / "output.tin(2).tif"    # 输入 GeoTIFF DEM（高程栅格）
    out = ROOT / "Dataset" / "output" / "map"               # 输出目录：保存 tiles/ 和索引文件
    tile = 256                                              # 输出 tile 的大小：tile x tile（像素）
    overlap = 0.0                                           # 重叠率：0=无重叠；0.5=50%重叠；越大样本越多
    k = 20                                                  # 代表性聚类簇数：最终大约选 k 张代表 tile
    seed = 0                                                # 随机种子：保证每次结果一致
    id = 20001                                              # 保存文件的id(起始记录编号)

def main():
    args = Args()

    os.makedirs(args.out, exist_ok=True) 
    tile_dir = os.path.join(args.out, "tiles")
    os.makedirs(tile_dir, exist_ok=True)
    png_dir = args.out / "tiles_png"
    png_dir.mkdir(parents=True, exist_ok=True)

    obstacle_dir = args.out / "hard_obstacle"
    obstacle_dir.mkdir(parents=True, exist_ok=True)

    soft_cost_dir = args.out / "soft_cost"
    soft_cost_dir.mkdir(parents=True, exist_ok=True)

    risk_png_dir = args.out / "risk_png"
    risk_png_dir.mkdir(parents=True, exist_ok=True)

    obstacle_png_dir = args.out / "obstacle_png"
    obstacle_png_dir.mkdir(parents=True, exist_ok=True)

    soft_cost_png_dir = args.out / "soft_cost_png"
    soft_cost_png_dir.mkdir(parents=True, exist_ok=True)

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

            nav_maps = build_nav_maps_from_dem(
                tile=tile
            )

            hard_obstacle = nav_maps["hard_obstacle"]          # uint8, 0/255
            soft_cost = nav_maps["soft_cost"]                  # float32, 0~1
            risk_u8 = nav_maps["risk_u8"]                      # uint8, 0~255
            grad_u8 = nav_maps["grad_u8"]                      # uint8, 0~255

            # 保存硬障碍图（建议 0/1 存npy，后面更方便）
            np.save(obstacle_dir / f"tile_{tile_id:05d}.npy", (hard_obstacle > 0).astype(np.uint8))

            # 保存软代价图
            np.save(soft_cost_dir / f"tile_{tile_id:05d}.npy", soft_cost.astype(np.float32))

            # 保存可视化 png
            imageio.imwrite(obstacle_png_dir / f"tile_{tile_id:05d}.png", 255-hard_obstacle)
            imageio.imwrite(risk_png_dir / f"tile_{tile_id:05d}.png", 255-risk_u8)
            imageio.imwrite(
                soft_cost_png_dir / f"tile_{tile_id:05d}.png",
                255-(soft_cost * 255.0).astype(np.uint8)
            )

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

def build_nav_maps_from_dem(
    tile: np.ndarray,              # 输入的一张 DEM 子图（二维高程栅格），shape 一般是 [tile, tile]
    pixel_size: float = 1.0,       # 每个像素对应的实际尺寸（米/像素）；用于把梯度换算成“每米高程变化”
    blur_ksize: int = 5,           # 高斯平滑核大小；越大越平滑，噪声越少，但细节也会被抹掉（必须是奇数）
    grad_w: float = 0.3,           # 梯度项权重；控制风险图中“坡度/边界强度”占多大比例
    lap_w: float = 0.1,           # Laplacian 项权重；控制风险图中“局部突变/曲率”占多大比例
    obstacle_percentile: float = 60.0,  # 硬障碍分位数阈值；表示把风险值最高的前 30% 区域切成障碍
    min_component_area: int = 20,   # 最小障碍连通域面积；小于该面积的碎小障碍块会被删除
    safe_margin_kernel: int = 3,   # 障碍膨胀核大小；用于给障碍外侧加安全边界
    safe_margin_iter: int = 1,     # 障碍膨胀次数；0 表示不扩张，1/2 表示逐步加粗障碍边界
):
    """
    输入:
        tile: 2D DEM tile, float32
        pixel_size: 栅格分辨率（米/像素）
    输出:
        dict:
            grad_u8         梯度图(0~255)
            risk_u8         融合风险图(0~255)
            hard_obstacle   硬障碍图(uint8, 0/255)
            soft_cost       软代价图(float32, 0~1)
    """
    t = tile.astype(np.float32)

    # 1) 先轻微平滑，减少噪声
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blur = cv2.GaussianBlur(t, (blur_ksize, blur_ksize), 0)

    # 2) Sobel 梯度：近似坡度 / 边界强度
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3) / max(pixel_size, 1e-6)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3) / max(pixel_size, 1e-6)
    grad = cv2.magnitude(gx, gy)
    grad_n = cv2.normalize(grad, None, 0, 1.0, cv2.NORM_MINMAX)

    # 3) Laplacian：近似曲率/突变程度
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)
    lap_n = cv2.normalize(lap_abs, None, 0, 1.0, cv2.NORM_MINMAX)

    # 4) 融合成 risk map
    risk = grad_w * grad_n + lap_w * lap_n
    risk = np.clip(risk, 0.0, 1.0)

    grad_u8 = (grad_n * 255.0).astype(np.uint8)
    risk_u8 = (risk * 255.0).astype(np.uint8)

    # 5) 用分位数阈值切出“硬障碍”
    # obstacle_percentile 越低 -> 障碍越多 -> 地图越难
    thr = np.percentile(risk_u8, obstacle_percentile)
    _, hard = cv2.threshold(risk_u8, float(thr), 255, cv2.THRESH_BINARY)

    # 6) 形态学清理：去噪点、补裂缝
    kernel3 = np.ones((3, 3), np.uint8)
    hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, kernel3)
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, kernel3)

    # 7) 去掉过小的障碍碎块
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    hard_filtered = np.zeros_like(hard)
    for i in range(1, num_labels):  # 0 是背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            hard_filtered[labels == i] = 255

    # 8) 给障碍留安全边界，避免“擦边走”
    if safe_margin_kernel > 1 and safe_margin_iter > 0:
        kernel_safe = np.ones((safe_margin_kernel, safe_margin_kernel), np.uint8)
        hard_filtered = cv2.dilate(hard_filtered, kernel_safe, iterations=safe_margin_iter)

    # 9) 距离变换 -> 软代价图
    # 障碍越近，代价越高；障碍内 cost=1
    free = (255 - hard_filtered).astype(np.uint8)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    soft_cost = 1.0 - dist
    soft_cost[hard_filtered > 0] = 1.0
    soft_cost = soft_cost.astype(np.float32)

    return {
        "grad_u8": grad_u8,
        "risk_u8": risk_u8,
        "hard_obstacle": hard_filtered,
        "soft_cost": soft_cost,
    }

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
