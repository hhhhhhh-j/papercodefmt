import numpy as np
import laspy
import rasterio
from rasterio.transform import from_origin
from rasterio.fill import fillnodata


def laz_to_dtm_tif(
    laz_path,
    out_tif,
    res=1.0,
    nodata=-9999.0,
    ground_class=2,
    fill_holes=True,
    max_search_distance=3
):
    # 1) 打开点云
    with laspy.open(laz_path) as f:
        header = f.header
        xmin, ymin, _ = header.mins
        xmax, ymax, _ = header.maxs

        width = int(np.ceil((xmax - xmin) / res))
        height = int(np.ceil((ymax - ymin) / res))

        # 初始化栅格：先放 inf，后面每格取最小 z
        grid = np.full((height, width), np.inf, dtype=np.float32)

        has_classification = "classification" in list(f.header.point_format.dimension_names)

        # 2) 分块读取，避免内存爆
        for points in f.chunk_iterator(2_000_000):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float32)

            # 3) 仅保留 ground 点
            if has_classification:
                cls = points.classification
                mask = (cls == ground_class)
                x = x[mask]
                y = y[mask]
                z = z[mask]

            if len(z) == 0:
                continue

            # 4) 映射到像元坐标
            col = np.floor((x - xmin) / res).astype(np.int64)
            row = np.floor((ymax - y) / res).astype(np.int64)

            valid = (
                (row >= 0) & (row < height) &
                (col >= 0) & (col < width)
            )
            row = row[valid]
            col = col[valid]
            z = z[valid]

            if len(z) == 0:
                continue

            # 5) 每个格子取最小 z，近似 DTM
            flat_idx = row * width + col
            order = np.argsort(flat_idx)
            flat_idx = flat_idx[order]
            z = z[order]

            uniq_idx, starts = np.unique(flat_idx, return_index=True)
            z_min = np.minimum.reduceat(z, starts)

            rr = uniq_idx // width
            cc = uniq_idx % width

            # 与已有值比较，再取更小的
            old_vals = grid[rr, cc]
            grid[rr, cc] = np.minimum(old_vals, z_min)

        # 6) 把没有值的地方设成 nodata
        mask_valid = np.isfinite(grid)
        grid[~mask_valid] = nodata

        # 7) 可选：对小孔洞做填补
        if fill_holes:
            valid_mask = (grid != nodata).astype(np.uint8)
            filled = fillnodata(
                grid,
                mask=valid_mask,
                max_search_distance=max_search_distance
            )
            filled[filled == 0] = nodata
            grid = filled.astype(np.float32)

        # 8) 写 GeoTIFF
        transform = from_origin(xmin, ymax, res, res)
        crs = header.parse_crs()

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": crs,
            "transform": transform,
            "nodata": nodata,
            "compress": "deflate"
        }

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(grid, 1)

    print(f"Saved: {out_tif}")

def main():
    laz_to_dtm_tif(
        laz_path="Dataset/point_cloud/points.laz",
        out_tif="Dataset/dem/dtm.tif",
        res=1.0,
        nodata=-9999.0,
        ground_class=2,
        fill_holes=True,
        max_search_distance=3
    )

if __name__ == "__main__":    main()