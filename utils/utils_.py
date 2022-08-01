from __future__ import absolute_import
import os
import sys
import errno
<<<<<<< HEAD
=======
import shutil
>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad
import json
import os.path as osp

import torch


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    """Computes and stores the average and current value.

<<<<<<< HEAD
    Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
=======
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


<<<<<<< HEAD
def save_checkpoint(state, fpath="checkpoint.pth.tar"):
=======
def save_checkpoint(state, fpath='checkpoint.pth.tar'):
>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
<<<<<<< HEAD
            self.file = open(fpath, "w")
=======
            self.file = open(fpath, 'w')
>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def read_json(fpath):
<<<<<<< HEAD
    with open(fpath, "r") as f:
=======
    with open(fpath, 'r') as f:
>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
<<<<<<< HEAD
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))
=======
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))






>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad
