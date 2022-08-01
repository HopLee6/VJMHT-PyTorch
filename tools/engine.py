# hierachical co-transformer with cluster

import os
import h5py
import glob
import json
import random
import importlib
import numpy as np
import torch
from utils.sys_utils import *
from utils.vsum_tools import *
from tools.funcs import *


class sumnet:
    def __init__(self, hps):
        self.hps = hps
        self.model = None
        self.log_file = None
        self.verbose = hps.verbose

    def fix_keys(self, keys, dataset_name=None):
        """
        :param keys:
        :return:
        """
        # dataset_name = None
        if len(self.datasets) == 1:
            dataset_name = next(iter(self.datasets))

        keys_out = []
        for key in keys:
            t = key.split("/")
            if len(t) != 2:
                assert (
                    dataset_name is not None
                ), "ERROR dataset name in some keys is missing but there are multiple dataset {} to choose from".format(
                    len(self.datasets)
                )

                key_name = dataset_name + "/" + key
                keys_out.append(key_name)
            else:
                keys_out.append(key)

        return keys_out

    def load_datasets(self, datasets=None):
        """
        Loads all h5 datasets from the datasets list into a dictionary self.dataset
        referenced by their base filename
        :param datasets:  List of dataset filenames
        :return:
        """
        if datasets is None:
            datasets = self.hps.datasets

        datasets_dict = {}
        for dataset in datasets:
            _, base_filename = os.path.split(dataset)
            base_filename, _ = os.path.splitext(base_filename)
            print("Loading:", dataset)
            datasets_dict[base_filename] = h5py.File(dataset, "r")

        self.datasets = datasets_dict
        return datasets_dict

    def load_split_file(self, splits_file):

        self.dataset_name, self.dataset_type, self.splits = parse_splits_filename(
            splits_file
        )
        n_folds = len(self.splits)
        self.split_file = splits_file
        print("Loading splits from: ", splits_file)

        return n_folds

    def select_split(self, split_id):
        print("Selecting split: ", split_id)

        self.split_id = split_id
        n_folds = len(self.splits)
        assert self.split_id < n_folds, "split_id (got {}) exceeds {}".format(
            self.split_id, n_folds
        )

        split = self.splits[self.split_id]
        self.train_keys = split["train_keys"]
        self.test_keys = split["test_keys"]

        dataset_filename = self.hps.get_dataset_by_name(self.dataset_name)[0]
        _, dataset_filename = os.path.split(dataset_filename)
        dataset_filename, _ = os.path.splitext(dataset_filename)
        self.train_keys = self.fix_keys(self.train_keys, dataset_filename)
        self.test_keys = self.fix_keys(self.test_keys, dataset_filename)
        return

    def load_model(self, model_filename):
        self.model.load_state_dict(
            torch.load(model_filename, map_location=lambda storage, loc: storage)
        )
        return

    def initialize(self, cuda_device=None):
        rnd_seed = self.hps.rnd_seed
        random.seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)
        model_module = importlib.import_module("models." + self.hps.model_name)
        self.model = model_module.DSN(self.hps)
        self.model.eval()
        self.model.apply(weights_init)

        cuda_device = cuda_device or self.hps.cuda_device

        if self.hps.use_cuda:
            print("Setting CUDA device: ", cuda_device)
            torch.cuda.set_device(cuda_device)
            torch.cuda.manual_seed(rnd_seed)

        if self.hps.use_cuda:
            self.model.cuda()

        return

    def get_data(self, key):
        key_parts = key.split("/")
        assert len(key_parts) == 2, "ERROR. Wrong key name: " + key
        dataset, key = key_parts
        # if not key in self.datasets[dataset].keys():
        #     pdb.set_trace()
        return self.datasets[dataset][key]

    def lookup_weights_file(self, data_path):
        dataset_type_str = "" if self.dataset_type == "" else self.dataset_type + "_"
        weights_filename = data_path + "/models/{}_{}splits_{}.tar.pth".format(
            self.dataset_name, dataset_type_str, self.split_id
        )
        weights_filename = glob.glob(weights_filename)
        if len(weights_filename) == 0:
            print("Couldn't find model weights: ", weights_filename)
            return ""

        # Get the first weights filename in the dir
        weights_filename = weights_filename[0]

        return weights_filename

    def preprocess(self, key):
        dataset = self.get_data(key)
        seq = dataset["features"][...]  # sequence of features, (seq_len, dim)
        gtscore = dataset["gtscore"][...]
        gtscore -= gtscore.min()
        gtscore /= gtscore.max()

        if "change_points" in dataset.keys():
            boundary = dataset["change_points"][...]
        else:
            l_shot = 60
            n_frames = len(seq) * 15
            boundary = [
                [i * l_shot, (i + 1) * l_shot - 1]
                for i in range(int(np.ceil(n_frames / l_shot)))
            ]
            boundary[-1][-1] = n_frames - 1
            boundary = np.array(boundary)
        shots = []
        for i in range(len(boundary)):
            start = int(boundary[i, 0] / 15)
            end = int(boundary[i, 1] / 15) + 1
            shot = seq[start:end, :]
            shot = torch.from_numpy(shot)
            if self.hps.use_cuda:
                shot = shot.cuda()
            shots.append(shot)
        return {
            "shots": shots,
            "gtscore": gtscore,
            "boundary": boundary,
            "n_seq": len(gtscore),
        }

    def train(self, output_dir="EX-0"):

        print("Initializing model and optimizer...")

        criterion = torch.nn.MSELoss()

        if self.hps.use_cuda:
            criterion = criterion.cuda()

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(
            parameters, lr=self.hps.lr[0], weight_decay=self.hps.l2_req
        )

        print("Starting training...")

        max_val_fscore = 0
        max_val_fscore_epoch = 0
        train_keys = self.train_keys[:]

        cluster_res_file = "cluster.json"
        with open(cluster_res_file, "r") as f:
            cluster = json.load(f)

        temp = {}
        for key, value in cluster.items():
            temp[key] = [v for v in value if v in train_keys]

        vid2cls = {}
        for key, value in cluster.items():
            for v in value:
                vid2cls[v] = key

        for epoch in range(self.hps.epochs_max):
            self.model.train()

            print(
                "Epoch: {0:6}".format(str(epoch) + "/" + str(self.hps.epochs_max)),
                end="",
            )
            avg_loss = []

            random.shuffle(train_keys)

            for idx, key in enumerate(train_keys):
                c = vid2cls[key]
                inter_cls_vid = temp[c]
                n = len(inter_cls_vid)

                if n == 0:
                    key2 = key
                else:
                    if self.hps.rand_choice:
                        key2 = random.choice(inter_cls_vid)
                    else:
                        i = (inter_cls_vid.index(key) + 1) % n
                        key2 = inter_cls_vid[i]

                inputs1 = self.preprocess(key)
                inputs2 = self.preprocess(key2)
                probs1, probs2, rec_loss = self.model(inputs1, inputs2)
                gtscore1 = inputs1["gtscore"]
                gtscore2 = inputs2["gtscore"]

                cost = (
                    criterion(probs1, torch.from_numpy(gtscore1).cuda())
                    + criterion(probs2, torch.from_numpy(gtscore2).cuda())
                ) / 2 * self.hps.supervised_factor + rec_loss * self.hps.rec_factor

                if self.hps.regular:
                    cost += (
                        0.1 * (probs1.mean() - 0.5) ** 2
                        + 0.1 * (probs2.mean() - 0.5) ** 2
                    ) / 2

                self.optimizer.zero_grad()

                cost.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                avg_loss.append(float(cost))

            # Evaluate test dataset
            if (epoch + 1) % self.hps.eval_freq == 0:
                val_fscore, video_scores = self.eval(self.test_keys)
                if max_val_fscore < val_fscore:
                    max_val_fscore = val_fscore
                    max_val_fscore_epoch = epoch
                print(
                    "   Test F-score avg/max: {0:0.5}/{1:0.5}".format(
                        val_fscore, max_val_fscore
                    )
                )

            avg_loss = np.array(avg_loss)
            print("   Train loss: {0:.05f}\n".format(np.mean(avg_loss)), end="")

            if self.verbose:
                video_scores = [["No", "Video", "F-score"]] + video_scores
                print_table(video_scores, cell_width=[3, 40, 8])

            # Save model weights
            if (epoch + 1) % self.hps.eval_freq == 0:
                path, filename = os.path.split(self.split_file)
                base_filename, _ = os.path.splitext(filename)
                path = os.path.join(
                    output_dir, "models_temp", base_filename + "_" + str(self.split_id)
                )
                os.makedirs(path, exist_ok=True)
                filename = (
                    str(epoch) + "_" + str(round(val_fscore * 100, 3)) + ".pth.tar"
                )
                torch.save(self.model.state_dict(), os.path.join(path, filename))

        return max_val_fscore, max_val_fscore_epoch

    def eval(self, keys, results_filename=None):

        self.model.eval()
        summary = {}
        att_vecs = {}
        with torch.no_grad():
            for i, key in enumerate(keys):
                inputs = self.preprocess(key)
                probs = self.model(inputs)
                probs = probs.data.cpu().numpy()
                summary[key] = probs

        f_score, video_scores = self.eval_summary(
            summary,
            keys,
            metric=self.dataset_name,
            results_filename=results_filename,
        )

        return f_score, video_scores

    def eval_summary(
        self,
        machine_summary_activations,
        test_keys,
        results_filename=None,
        metric="tvsum",
    ):

        eval_metric = "avg" if metric == "tvsum" else "max"

        if results_filename is not None:
            h5_res = h5py.File(results_filename, "w")

        fms = []
        video_scores = []
        for key_idx, key in enumerate(test_keys):
            d = self.get_data(key)
            probs = machine_summary_activations[key]

            if "change_points" not in d:
                print("ERROR: No change points in dataset/video ", key)

            cps = d["change_points"][...]
            num_frames = d["n_frames"][()]
            nfps = d["n_frame_per_seg"][...].tolist()
            positions = d["picks"][...]
            user_summary = d["user_summary"][...]

            machine_summary = generate_summary(probs, cps, num_frames, nfps, positions)
            fm, _, _ = evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

            # Reporting & logging
            video_scores.append([key_idx + 1, key, "{:.1%}".format(fm)])

            if results_filename:
                gt = d["gtscore"][...]
                h5_res.create_dataset(key + "/score", data=probs)
                h5_res.create_dataset(key + "/machine_summary", data=machine_summary)
                h5_res.create_dataset(key + "/gtscore", data=gt)
                h5_res.create_dataset(key + "/fm", data=fm)
                h5_res.create_dataset(key + "/picks", data=positions)

                video_name = key.split("/")[1]
                if "video_name" in d:
                    video_name = d["video_name"][...]
                h5_res.create_dataset(key + "/video_name", data=video_name)

        mean_fm = np.mean(fms)

        # Reporting & logging
        if results_filename is not None:
            h5_res.close()

        return mean_fm, video_scores
