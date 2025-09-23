import numpy as np
import os
from torch.utils.data import Dataset
import torch
from pointnet_util import farthest_point_sample, pc_normalize

def create_zigzag_paths_3d(points, num_paths=6):
    N = points.shape[0] 
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

class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000, use_zigzag=False, num_paths=6, zigzag_indices=None):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel
      
        self.use_zigzag = use_zigzag
        self.num_paths = num_paths
        self.zigzag_indices = zigzag_indices if zigzag_indices else [0, 1, 2]
      
        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))
        self.cache_size = cache_size 
        self.cache = {}  
        self.zigzag_path_counter = 0
        self.original_order_counter = 0
        self.total_samples = 0
        self.path_usage = {}  
        self.split = split

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
          
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]
          
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
          
            if not self.normal_channel:
                point_set = point_set[:, 0:3]
          
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)
      
        self.total_samples += 1
      
        if self.use_zigzag:
            paths, path_indices = create_zigzag_paths_3d(point_set[:, 0:3], num_paths=self.num_paths)
            valid_indices = [idx for idx in self.zigzag_indices if idx < len(paths)]
            if not valid_indices:
                valid_indices = [0] if len(paths) > 0 else []
            if self.split == 'train':
                if valid_indices:
                    path_idx = np.random.choice(valid_indices)
                    current_path = paths[path_idx]
                    self.zigzag_path_counter += 1
                    if path_idx not in self.path_usage:
                        self.path_usage[path_idx] = 0
                    self.path_usage[path_idx] += 1
                else:
                    current_path = np.arange(len(point_set))
                    self.original_order_counter += 1
            else:
                if valid_indices:
                    path_idx = valid_indices[0]
                    current_path = paths[path_idx]
                    self.zigzag_path_counter += 1
                    if path_idx not in self.path_usage:
                        self.path_usage[path_idx] = 0
                    self.path_usage[path_idx] += 1
                else:
                    current_path = np.arange(len(point_set))
                    self.original_order_counter += 1
            point_set = point_set[current_path]
            if self.total_samples % 500 == 0:
                zigzag_percent = self.zigzag_path_counter / self.total_samples * 100
                original_percent = self.original_order_counter / self.total_samples * 100
                for p_idx, count in sorted(self.path_usage.items()):
                    path_percent = count / self.zigzag_path_counter * 100 if self.zigzag_path_counter > 0 else 0
      
        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)

class PartNormalDataset(Dataset):
    def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', 
                 class_choice=None, normal_channel=False, use_zigzag=False, num_paths=6, zigzag_indices=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_zigzag = use_zigzag
        self.num_paths = num_paths
        self.zigzag_indices = zigzag_indices
        self.zigzag_path_counter = 0
        self.original_order_counter = 0
        self.total_samples = 0
        self.path_usage = {} 
        self.split = split
      
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}

        import json
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))

            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.cache = {}  
        self.cache_size = 20000
      
        if self.use_zigzag:
            print(f'[DATASET] Using zigzag path ordering with {self.num_paths} paths')

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
          
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
              
            seg = data[:, -1].astype(np.int32)
          
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        self.total_samples += 1
      
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]
        if self.use_zigzag:
            paths, path_indices = create_zigzag_paths_3d(point_set[:, 0:3], num_paths=self.num_paths)
            valid_indices = [idx for idx in self.zigzag_indices if idx < len(paths)]
            if not valid_indices:
                valid_indices = [0] if len(paths) > 0 else []
            if self.split == 'train' or self.split == 'trainval':
                if valid_indices:
                    path_idx = np.random.choice(valid_indices)
                    current_path = paths[path_idx]
                    self.zigzag_path_counter += 1
                    if path_idx not in self.path_usage:
                        self.path_usage[path_idx] = 0
                    self.path_usage[path_idx] += 1
                else:
                    current_path = np.arange(len(point_set))
                    self.original_order_counter += 1
            else:
                if valid_indices:
                    path_idx = valid_indices[0]
                    current_path = paths[path_idx]
                    self.zigzag_path_counter += 1
                    if path_idx not in self.path_usage:
                        self.path_usage[path_idx] = 0
                    self.path_usage[path_idx] += 1
                else:
                    current_path = np.arange(len(point_set))
                    self.original_order_counter += 1
            point_set = point_set[current_path]
            seg = seg[current_path]
          
            if self.total_samples % 500 == 0:
                zigzag_percent = self.zigzag_path_counter / self.total_samples * 100
                original_percent = self.original_order_counter / self.total_samples * 100
                for p_idx, count in sorted(self.path_usage.items()):
                    path_percent = count / self.zigzag_path_counter * 100 if self.zigzag_path_counter > 0 else 0
      
        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    data = ModelNetDataLoader('modelnet40_normal_resampled/', split='train', uniform=False, normal_channel=True)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,label in DataLoader:
        print(point.shape)
        print(label.shape)
