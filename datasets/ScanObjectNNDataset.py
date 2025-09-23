# 3月23日——zigzag
import numpy as np
import os, sys, h5py
from torch.utils.data import Dataset
import torch
from .build import DATASETS
from utils.logger import *
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

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
class ScanObjectNN(Dataset):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.subset = config.subset
        self.root = config.ROOT
        
        self.use_zigzag = config.use_zigzag
        self.num_paths = config.num_paths
        self.zigzag_indices = config.zigzag_indices
        self.zigzag_path_counter = 0
        self.original_order_counter = 0
        self.total_samples = 0
        self.path_usage = {} 
        with open('/tmp/zigzag_debug_scanobjectnn.txt', 'w') as f:
            f.write(f"ScanObjectNN Dataset Initialization\n")
            f.write(f"use_zigzag: {self.use_zigzag}\n")
            f.write(f"num_paths: {self.num_paths}\n")
            f.write(f"zigzag_indices: {self.zigzag_indices}\n")
        
        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()
        
        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')
        
        if self.use_zigzag:
            print_log(f'[DATASET] Using zigzag path ordering with {self.num_paths} paths for ScanObjectNN', logger='ScanObjectNN')
        
    def __getitem__(self, idx):
        if idx == 0: 
            with open('/tmp/zigzag_debug_scanobjectnn.txt', 'a') as f:
                f.write(f"First sample accessed: use_zigzag={self.use_zigzag}, num_paths={self.num_paths}\n")
        self.total_samples += 1
        if self.use_zigzag:
            current_points = self.points[idx].copy()
            paths, path_indices = create_zigzag_paths_3d(current_points, num_paths=self.num_paths)
            for i in range(len(paths)):
                if len(paths[i]) != len(current_points):
                    print(f"Warning: Path {i} length ({len(paths[i])}) does not match data length ({len(current_points)})")
            
            valid_indices = [i for i in self.zigzag_indices if i < len(paths)]
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
                    current_path = np.arange(len(current_points))
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
                    current_path = np.arange(len(current_points))
                    self.original_order_counter += 1
                    path_info = "original_order"
            current_points = current_points[current_path]
            if self.total_samples % 100 == 0:
                zigzag_percent = self.zigzag_path_counter / self.total_samples * 100
                original_percent = self.original_order_counter / self.total_samples * 100
                for p_idx, count in sorted(self.path_usage.items()):
                    path_percent = count / self.zigzag_path_counter * 100 if self.zigzag_path_counter > 0 else 0

                if len(current_path) > 10:
                    original_indices = list(range(10))
                    current_indices = current_path[:10].tolist() if isinstance(current_path, np.ndarray) else current_path[:10]
                with open('/tmp/zigzag_path_stats_scanobjectnn.txt', 'a') as f:
                    f.write(f"\n\n========== ZigZag Path Usage Statistics ==========\n")
                    f.write(f"Total samples: {self.total_samples}\n")
                    f.write(f"ZigZag paths: {self.zigzag_path_counter} ({zigzag_percent:.2f}%)\n")
                    f.write(f"Original order: {self.original_order_counter} ({original_percent:.2f}%)\n")
                    f.write(f"Current sample ({idx}): Using {path_info}\n")
                    f.write(f"Path length: {len(current_path)}, Data length: {len(current_points)}\n")
                    
                    f.write(f"Path usage details:\n")
                    for p_idx, count in sorted(self.path_usage.items()):
                        path_percent = count / self.zigzag_path_counter * 100 if self.zigzag_path_counter > 0 else 0
                        f.write(f"  Path {p_idx}: {count} ({path_percent:.2f}%)\n")
        else:
            pt_idxs = np.arange(0, self.points.shape[1])  # 2048
            if self.subset == 'train':
                np.random.shuffle(pt_idxs)
            current_points = self.points[idx, pt_idxs].copy()
        
        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]
        return 'ScanObjectNN', 'sample', (current_points, label)
    
    def __len__(self):
        return self.points.shape[0]

@DATASETS.register_module()
class ScanObjectNN_hardest(Dataset):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.subset = config.subset
        self.root = config.ROOT
        
        self.use_zigzag = hasattr(config, 'use_zigzag') and config.use_zigzag
        self.num_paths = config.num_paths if hasattr(config, 'num_paths') else 9
        self.zigzag_indices = [0, 1, 2] if not hasattr(config, 'zigzag_indices') else config.zigzag_indices
        
        self.zigzag_path_counter = 0
        self.original_order_counter = 0
        self.total_samples = 0
        self.path_usage = {}
        with open('/tmp/zigzag_debug_scanobjectnn_hardest.txt', 'w') as f:
            f.write(f"ScanObjectNN_hardest Dataset Initialization\n")
            f.write(f"use_zigzag: {self.use_zigzag}\n")
            f.write(f"num_paths: {self.num_paths}\n")
            f.write(f"zigzag_indices: {self.zigzag_indices}\n")
        
        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()
        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')
        if self.use_zigzag:
            print_log(f'[DATASET] Using zigzag path ordering with {self.num_paths} paths for ScanObjectNN_hardest', logger='ScanObjectNN')
    
    def __getitem__(self, idx):
        if idx == 0:
            with open('/tmp/zigzag_debug_scanobjectnn_hardest.txt', 'a') as f:
                f.write(f"First sample accessed: use_zigzag={self.use_zigzag}, num_paths={self.num_paths}\n")
        self.total_samples += 1
        if self.use_zigzag:
            current_points = self.points[idx].copy()
            paths, path_indices = create_zigzag_paths_3d(current_points, num_paths=self.num_paths)
            for i in range(len(paths)):
                if len(paths[i]) != len(current_points):
                    print(f"Warning: Path {i} length ({len(paths[i])}) does not match data length ({len(current_points)})")
            
            valid_indices = [i for i in self.zigzag_indices if i < len(paths)]
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
                    current_path = np.arange(len(current_points))
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
                    current_path = np.arange(len(current_points))
                    self.original_order_counter += 1
                    path_info = "original_order"
            current_points = current_points[current_path]
            if self.total_samples % 100 == 0:
                zigzag_percent = self.zigzag_path_counter / self.total_samples * 100
                original_percent = self.original_order_counter / self.total_samples * 100
                for p_idx, count in sorted(self.path_usage.items()):
                    path_percent = count / self.zigzag_path_counter * 100 if self.zigzag_path_counter > 0 else 0
                if len(current_path) > 10:
                    original_indices = list(range(10))
                    current_indices = current_path[:10].tolist() if isinstance(current_path, np.ndarray) else current_path[:10]
                with open('/tmp/zigzag_path_stats_scanobjectnn_hardest.txt', 'a') as f:
                    f.write(f"\n\n========== ZigZag Path Usage Statistics ==========\n")
                    f.write(f"Total samples: {self.total_samples}\n")
                    f.write(f"ZigZag paths: {self.zigzag_path_counter} ({zigzag_percent:.2f}%)\n")
                    f.write(f"Original order: {self.original_order_counter} ({original_percent:.2f}%)\n")
                    f.write(f"Current sample ({idx}): Using {path_info}\n")
                    f.write(f"Path length: {len(current_path)}, Data length: {len(current_points)}\n")
                    f.write(f"Path usage details:\n")
                    for p_idx, count in sorted(self.path_usage.items()):
                        path_percent = count / self.zigzag_path_counter * 100 if self.zigzag_path_counter > 0 else 0
                        f.write(f"  Path {p_idx}: {count} ({path_percent:.2f}%)\n")
        else:
            pt_idxs = np.arange(0, self.points.shape[1])  # 2048
            if self.subset == 'train':
                np.random.shuffle(pt_idxs)
            current_points = self.points[idx, pt_idxs].copy()
        
        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]
        return 'ScanObjectNN', 'sample', (current_points, label)
    
    def __len__(self):
        return self.points.shape[0]


