# 构建topo map加入到状态空间
# 需要一个节点类 Node
# 以及一个topo map类 TopoMap
import numpy as np
import cv2

# 定义三种状态的描述
UNKNOWN = -1
FREE    = 0
OCC     = 1

class Frontier:
    def __init__(self, local_occ, local_unk, local_free, robot_ij, k=8):
        self.k = k  # 选择前 k 个 frontier clusters
        H, W = local_free.shape
        self.grid = np.full((H, W), UNKNOWN, dtype=np.int8)
        self.grid[local_occ >= 0.7] = OCC
        self.grid[local_free >= 0.7] = FREE
        self.robot_ij = robot_ij  # 机器人在 grid 中的位置 (i,j)

    def compute_frontier_mask(self):
        free = (self.grid == FREE).astype(np.uint8)
        unk  = (self.grid == UNKNOWN).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        unk_near = cv2.dilate(unk, kernel, iterations=1)  # unknown 周围一圈

        frontier = (free & unk_near).astype(np.uint8)
        return frontier  # 0/1


    def cluster_frontiers(self, frontiers, min_size=3):
        # labels: 0=背景, 1..num_labels-1 = 每个连通域
        num_labels, labels = cv2.connectedComponents(frontiers, connectivity=8)

        clusters = []
        for lab in range(1, num_labels):
            ys, xs = np.where(labels == lab)   # ys=i, xs=j
            if ys.size < min_size:
                continue
            cells = np.stack([ys, xs], axis=1)  # (N,2) 每行 (i,j)
            clusters.append(cells)

        return clusters


    def summarize_clusters_rep_points(self, clusters):
        ri, rj = self.robot_ij
        infos = []

        for cells in clusters:
            # centroid (浮点)
            ic = float(cells[:, 0].mean())
            jc = float(cells[:, 1].mean())

            # rep：离机器人最近的格子
            di = cells[:, 0] - ri
            dj = cells[:, 1] - rj
            k = int(np.argmin(di*di + dj*dj))
            rep_ij = (int(cells[k, 0]), int(cells[k, 1]))

            infos.append({
                "cells": cells,              
                "size": int(cells.shape[0]),
                "centroid_ij": (ic, jc),
                "rep_ij": rep_ij
            })

        return infos

    def select_topk(self, infos, risk_map=None):
        ri, rj = self.robot_ij

        def score(info):
            i, j = info["rep_ij"]
            dist = np.hypot(i - ri, j - rj) + 1e-6
            size = info["size"]

            risk = 0.0
            if risk_map is not None:
                risk = float(risk_map[i, j])

            # 分数越大越好：偏好“大cluster、近一些、低风险”
            return 2.6 * size - 1.0 * dist - 2.0 * risk

        infos_sorted = sorted(infos, key=score, reverse=True)
        topk = infos_sorted[:self.k]

        # padding（不足 K 用 None 填满）
        while len(topk) < self.k:
            topk.append(None)

        return topk
        
