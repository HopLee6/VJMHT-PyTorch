import os
import glob
import json
import hdf5storage
import numpy as np
import torch.nn.init as init
from utils.sys_utils import *
from utils.vsum_tools import *


def get_vidnames(dset):
    if dset == "summe":
        path = "rethinking_data/data/raw/summe/GT"
        vids = sorted(os.listdir(path))
        name = [vid.replace(".mat", "") for vid in vids]
    if dset == "tvsum":
        filename = "rethinking_data/data/raw/tvsum/ydata-tvsum50.mat"
        data = hdf5storage.loadmat(filename, variable_names=["tvsum50"])
        data = data["tvsum50"].ravel()
        name = [i[0][0, 0] for i in data]
    return name


def weights_init(m):
    classname = m.__class__.__name__
    if classname == "Linear":
        init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
        if m.bias is not None:
            init.constant_(m.bias, 0.1)


def parse_splits_filename(splits_filename):
    # Parse split file and count number of k_folds
    spath, sfname = os.path.split(splits_filename)
    sfname, _ = os.path.splitext(sfname)
    dataset_name = sfname.split("_")[0]  # Get dataset name e.g. tvsum
    dataset_type = sfname.split("_")[1]  # augmentation type e.g. aug

    # The keyword 'splits' is used as the filename fields terminator from historical reasons.
    if dataset_type == "splits":
        # Split type is not present
        dataset_type = ""

    # Get number of discrete splits within each split json file
    with open(splits_filename, "r") as sf:
        splits = json.load(sf)

    return dataset_name, dataset_type, splits


def lookup_weights_splits_file(path, dataset_name, dataset_type, split_id):
    dataset_type_str = "" if dataset_type == "" else dataset_type + "_"
    weights_filename = path + "/models/{}_{}splits_{}_*.tar.pth".format(
        dataset_name, dataset_type_str, split_id
    )
    weights_filename = glob.glob(weights_filename)
    if len(weights_filename) == 0:
        print("Couldn't find model weights: ", weights_filename)
        return ""

    # Get the first weights file in the dir
    weights_filename = weights_filename[0]
    splits_file = path + "/splits/{}_{}splits.json".format(
        dataset_name, dataset_type_str
    )

    return weights_filename, splits_file
