from utils import registry
# from datasets.ModelNetDatasetFewShot import ModelNetFewShot

DATASETS = registry.Registry('dataset')


def build_dataset_from_cfg(cfg, default_args=None):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (eDICT): 
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    from .ModelNetDatasetFewShot import ModelNetFewShot  # 确保在 DATASETS 定义后导入
    print("Registered datasets:", DATASETS.module_dict.keys())  # 调试输出
    return DATASETS.build(cfg, default_args=default_args)

