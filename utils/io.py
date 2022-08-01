import hdf5storage
import json
<<<<<<< HEAD
=======
import tables
>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad
import numpy as np
import os
import scipy.io as sio
import json


def load_sumres(dirname):
<<<<<<< HEAD
    with open(dirname, "r") as f:
=======
    with open(dirname, 'r') as f:
>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad
        data = json.load(f)

    data_list = {}
    for key, val in data.items():
<<<<<<< HEAD
        data_list[key] = np.array([val["summary"]]).T
=======
        data_list[key] = np.array([val['summary']]).T
>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad
    return data_list


def load_summe_mat(dirname):
    mat_list = os.listdir(dirname)

    data_list = []
    for mat in mat_list:
        data = sio.loadmat(os.path.join(dirname, mat))

        item_dict = {
<<<<<<< HEAD
            "video": mat[:-4],
            "length": data["video_duration"],
            "nframes": data["nFrames"],
            "user_anno": data["user_score"],
            "gt_score": data["gt_score"],
=======
            'video': mat[:-4],
            'length': data['video_duration'],
            'nframes': data['nFrames'],
            'user_anno': data['user_score'],
            'gt_score': data['gt_score']
>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad
        }

        data_list.append((item_dict))

    return data_list


def load_tvsum_mat(filename):
<<<<<<< HEAD
    data = hdf5storage.loadmat(filename, variable_names=["tvsum50"])
    data = data["tvsum50"].ravel()
=======
    data = hdf5storage.loadmat(filename, variable_names=['tvsum50'])
    data = data['tvsum50'].ravel()
>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad

    data_list = []
    for item in data:
        video, category, title, length, nframes, user_anno, gt_score = item

        item_dict = {
<<<<<<< HEAD
            "video": video[0, 0],
            "category": category[0, 0],
            "title": title[0, 0],
            "length": length[0, 0],
            "nframes": nframes[0, 0],
            "user_anno": user_anno,
            "gt_score": gt_score,
=======
            'video': video[0, 0],
            'category': category[0, 0],
            'title': title[0, 0],
            'length': length[0, 0],
            'nframes': nframes[0, 0],
            'user_anno': user_anno,
            'gt_score': gt_score
>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad
        }

        data_list.append((item_dict))

<<<<<<< HEAD
    return data_list
=======
    return data_list
>>>>>>> d1a96e10480e3d10294c2ef1b61a8f5361e362ad
