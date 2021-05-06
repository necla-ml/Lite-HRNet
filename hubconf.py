import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmpose.apis import multi_gpu_test, single_gpu_test
from mmpose.core import wrap_fp16_model
from mmpose.datasets import build_dataloader, build_dataset
from models import build_posenet

ARCH = dict(litehrnet_18_coco_256x192='1ZewlvpncTvahbqcCFb-95C3NHet30mk5',
            litehrnet_18_coco_384x288='1E3S18YbUfBm7YtxYOV7I9FmrntnlFKCp',
            litehrnet_30_coco_256x192='1KLjNInzFfmZWSbEQwx-zbyaBiLB7SnEj',
            litehrnet_30_coco_384x288='1BcHnLka4FWiXRmPnJgJKmsSuXXqN4dgn',
            litehrnet_18_mpii_256x256='1bcnn5Ic2-FiSNqYOqLd1mOfQchAz_oCf',
            litehrnet_30_mpii_256x256='1JB9LOwkuz5OUtry0IQqXammFuCrGvlEd')

def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        default=None,
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1

def posenet(arch='litehrnet_30_coco_384x288'):
    assert arch in ARCH, f"{arch} not supported, try one of {ARCH}"
    from pathlib import Path
    CWD = Path(__file__).parent
    prefix = CWD / 'configs/top_down/lite_hrnet'
    dataset = 'coco' if 'coco' in arch else 'mpii'
    cfg = Config.fromfile(f"{prefix}/{dataset}/{arch}.py")
    model = build_posenet(cfg.model)
    model.cfg = cfg
    return model
