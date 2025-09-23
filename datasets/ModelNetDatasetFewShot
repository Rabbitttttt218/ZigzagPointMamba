import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import *
import torch
import random

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def create_zigzag_paths_3d(points, num_paths=9):
    N = points.shape[0]  # 点的总数
    paths = []
    path_indices = []

    num_xy_paths = num_paths // 3 + (num_paths % 3 > 0)
    z_sorted = np.argsort(points[:, 2])
    z_layers = np.array_split(z_sorted, num_xy_paths)

    for layer_idx, layer_points in enumerate(z_layers):
        if len(layer_points) == 0:
            continue

        layer_points_coords = points[layer_points]

        x_sorted = layer_points[np.argsort(layer_points_coords[:, 0])]

        x_segments = np.array_split(x_sorted, max(1, min(len(x_sorted) // 20, 10)))
        path = []

        for i, segment in enumerate(x_segments):
            if len(segment) == 0:
                continue

            segment_coords = points[segment]

            if i % 2 == 0:
                sorted_segment = segment[np.argsort(segment_coords[:, 1])]
            else:
                sorted_segment = segment[np.argsort(-segment_coords[:, 1])]

            path.extend(sorted_segment)

        if path:
            seen_indices = set(path)
            unseen_indices = list(set(range(N)) - seen_indices)

            if unseen_indices:
                path.extend(unseen_indices)

            paths.append(np.array(path))
            path_indices.append(len(paths) - 1)

    num_xz_paths = num_paths // 3 + (num_paths % 3 > 1)
    y_sorted = np.argsort(points[:, 1])
    y_layers = np.array_split(y_sorted, num_xz_paths)

    for layer_idx, layer_points in enumerate(y_layers):
        if len(layer_points) == 0:
            continue

        layer_points_coords = points[layer_points]

        x_sorted = layer_points[np.argsort(layer_points_coords[:, 0])]

        x_segments = np.array_split(x_sorted, max(1, min(len(x_sorted) // 20, 10)))
        path = []

        for i, segment in enumerate(x_segments):
            if len(segment) == 0:
                continue

            segment_coords = points[segment]

            if i % 2 == 0:
                sorted_segment = segment[np.argsort(segment_coords[:, 2])]
            else:
                sorted_segment = segment[np.argsort(-segment_coords[:, 2])]

            path.extend(sorted_segment)

        if path:
            seen_indices = set(path)
            unseen_indices = list(set(range(N)) - seen_indices)

            if unseen_indices:
                path.extend(unseen_indices)

            paths.append(np.array(path))
            path_indices.append(len(paths) - 1)

    num_yz_paths = num_paths // 3
    x_sorted = np.argsort(points[:, 0])
    x_layers = np.array_split(x_sorted, num_yz_paths)

    for layer_idx, layer_points in enumerate(x_layers):
        if len(layer_points) == 0:
            continue
        layer_points_coords = points[layer_points]

        y_sorted = layer_points[np.argsort(layer_points_coords[:, 1])]

        y_segments = np.array_split(y_sorted, max(1, min(len(y_sorted) // 20, 10)))
        path = []

        for i, segment in enumerate(y_segments):
            if len(segment) == 0:
                continue

            segment_coords = points[segment]

            if i % 2 == 0:
                sorted_segment = segment[np.argsort(segment_coords[:, 2])]
            else:
                sorted_segment = segment[np.argsort(-segment_coords[:, 2])]

            path.extend(sorted_segment)

        if path:
            seen_indices = set(path)
            unseen_indices = list(set(range(N)) - seen_indices)

            if unseen_indices:
                path.extend(unseen_indices)

            paths.append(np.array(path))
            path_indices.append(len(paths) - 1)

    if not paths:
        paths.append(np.arange(N))
        path_indices.append(0)

    return paths, path_indices


@DATASETS.register_module()
class ModelNetFewShot(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        split = config.subset
        self.subset = config.subset

        self.way = config.way
        self.shot = config.shot
        self.fold = config.fold
        if self.way == -1 or self.shot == -1 or self.fold == -1:
            raise RuntimeError()

        self.pickle_path = os.path.join(self.root, f'{self.way}way_{self.shot}shot', f'{self.fold}.pkl')

        self.use_zigzag = config.use_zigzag
        self.num_paths = config.num_paths
        self.zigzag_indices = config.zigzag_indices

        self.zigzag_path_counter = 0
        self.original_order_counter = 0
        self.total_samples = 0
        self.path_usage = {} 

        print_log('Load processed data from %s...' % self.pickle_path, logger='ModelNetFewShot')

        with open(self.pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)[self.subset]

        print_log('The size of %s data is %d' % (split, len(self.dataset)), logger='ModelNetFewShot')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        points, label, _ = self.dataset[index]

        points[:, 0:3] = pc_normalize(points[:, 0:3])
        if not self.use_normals:
            points = points[:, 0:3]

        if self.use_zigzag:
            paths, path_indices = create_zigzag_paths_3d(points, num_paths=self.num_paths)

            for i in range(len(paths)):
                if len(paths[i]) != len(points):
                    print(f"Warning: Path {i} length ({len(paths[i])}) does not match data length ({len(points)})")

            valid_indices = [idx for idx in self.zigzag_indices if idx < len(paths)]
            if not valid_indices:
                valid_indices = [0] if len(paths) > 0 else []
            path_info = ""
            if self.subset == 'train':
                if valid_indices:
                    path_idx = np.random.choice(valid_indices)
                    current_path = paths[path_idx]
                    self.zigzag_path_counter += 1

                    if path_idx not in self.path_usage:
                        self.path_usage[path_idx] = 0
                    self.path_usage[path_idx] += 1

                    path_info = f"zigzag_path_{path_idx}"
                else:
                    current_path = np.arange(len(points))
                    self.original_order_counter += 1
                    path_info = "original_order"
            else:
                if valid_indices:
                    path_idx = valid_indices[0]
                    current_path = paths[path_idx]
                    self.zigzag_path_counter += 1

                    if path_idx not in self.path_usage:
                        self.path_usage[path_idx] = 0
                    self.path_usage[path_idx] += 1

                    path_info = f"zigzag_path_{path_idx}"
                else:
                    current_path = np.arange(len(points))
                    self.original_order_counter += 1
                    path_info = "original_order"

            current_points = points[current_path].copy()
        else:
            pt_idxs = np.arange(0, points.shape[0])
            if self.subset == 'train':
                np.random.shuffle(pt_idxs)
            current_points = points[pt_idxs].copy()
        if self.total_samples % 100 == 0 and self.use_zigzag:
            with open('/tmp/zigzag_path_stats_modelnet.txt', 'a') as f:
                f.write(f"\n\n========== ZigZag Path Usage Statistics ==========\n")

        current_points = torch.from_numpy(current_points).float()
        return 'ModelNet', 'sample', (current_points, label)
