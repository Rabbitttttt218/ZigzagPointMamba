import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *

def create_zigzag_paths_3d(points, num_paths=6):
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
class ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        self.sample_points_num = config.npoints
        self.whole = config.get('whole')
        self.use_zigzag = config.use_zigzag
        self.num_paths = config.num_paths
        self.zigzag_indices = config.zigzag_indices
        self.zigzag_path_counter = 0
        self.original_order_counter = 0
        self.total_samples = 0
        self.path_usage = {}
        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                  test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger='ShapeNet-55')
            lines = test_lines + lines
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='ShapeNet-55')
        self.permutation = np.arange(self.npoints)
        if self.use_zigzag:
            print_log(f'[DATASET] Using zigzag path ordering with {self.num_paths} paths', logger='ShapeNet-55')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
    
    def __getitem__(self, idx):
        self.total_samples += 1
        
        sample = self.file_list[idx]
        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        data = self.random_sample(data, self.sample_points_num)
        
        # 归一化点云
        data = self.pc_norm(data)
        if self.use_zigzag:
            paths, path_indices = create_zigzag_paths_3d(data, num_paths=self.num_paths)
            for i in range(len(paths)):
                if len(paths[i]) != len(data):
                    print(f"Warning: Path {i} length ({len(paths[i])}) does not match data length ({len(data)})")
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
                    current_path = np.arange(len(data))
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
                    current_path = np.arange(len(data))
                    self.original_order_counter += 1
                    path_info = "original_order"
            data = data[current_path]
            if self.total_samples % 100 == 0:
                zigzag_percent = self.zigzag_path_counter / self.total_samples * 100
                original_percent = self.original_order_counter / self.total_samples * 100
                for p_idx, count in sorted(self.path_usage.items()):
                    path_percent = count / self.zigzag_path_counter * 100 if self.zigzag_path_counter > 0 else 0
                if len(current_path) > 10:
                    original_indices = list(range(10))
                    current_indices = current_path[:10].tolist() if isinstance(current_path, np.ndarray) else current_path[:10]
        data = torch.from_numpy(data).float()
        
        return sample['taxonomy_id'], sample['model_id'], data

    def __len__(self):
        return len(self.file_list)
