import argparse
from utils.sys_utils import *
from utils.vsum_tools import *
from tools.funcs import *
import importlib
import time


def eval_split(hps, splits_filename, data_dir="test"):

    print("\n")
    model = engine.sumnet(hps)
    model.initialize()
    model.load_datasets()
    model.load_split_file(splits_filename)

    val_fscores = []
    for split_id in range(len(model.splits)):
        model.select_split(split_id)
        weights_filename = model.lookup_weights_file(data_dir)
        print("Loading model:", weights_filename)
        model.load_model(weights_filename)
        val_fscore, video_scores = model.eval(model.test_keys)
        val_fscores.append(val_fscore)
        val_fscore_avg = np.mean(val_fscores)

        if hps.verbose:
            video_scores = [["No.", "Video", "F-score"]] + video_scores
            print_table(video_scores, cell_width=[4, 45, 5])

        print("Avg F-score: ", val_fscore)
        print("")

    print("Total AVG F-score: ", val_fscore_avg)
    return val_fscore_avg


def train(hps):
    os.makedirs(hps.output_dir, exist_ok=True)
    os.makedirs(os.path.join(hps.output_dir, "models"), exist_ok=True)

    # Create a file to collect results from all splits
    f = open(hps.output_dir + "/results.txt", "wt")

    for split_filename in hps.splits:
        dataset_name, dataset_type, splits = parse_splits_filename(split_filename)

        # For no augmentation use only a dataset corresponding to the split file
        datasets = None
        if dataset_type == "":
            datasets = hps.get_dataset_by_name(dataset_name)

        if datasets is None:
            datasets = hps.datasets

        f_avg = 0
        n_folds = len(splits)
        for split_id in range(n_folds):
            model = engine.sumnet(hps)
            model.initialize()
            model.load_datasets(datasets=datasets)
            model.load_split_file(splits_file=split_filename)
            model.select_split(split_id=split_id)

            fscore, fscore_epoch = model.train(output_dir=hps.output_dir)
            f_avg += fscore

            # Log F-score for this split_id
            f.write(
                split_filename
                + ", "
                + str(split_id)
                + ", "
                + str(fscore)
                + ", "
                + str(fscore_epoch)
                + "\n"
            )
            f.flush()

            # Save model with the highest F score
            _, log_file = os.path.split(split_filename)
            log_dir, _ = os.path.splitext(log_file)
            log_dir += "_" + str(split_id)
            log_file = os.path.join(hps.output_dir, "models", log_dir) + ".tar.pth"

            os.makedirs(
                os.path.join(
                    hps.output_dir,
                    "models",
                ),
                exist_ok=True,
            )
            os.system(
                "mv "
                + hps.output_dir
                + "/models_temp/"
                + log_dir
                + "/"
                + str(fscore_epoch)
                + "_*.pth.tar "
                + log_file
            )
            os.system("rm -rf " + hps.output_dir + "/models_temp")

            print(
                "Split: {0:}   Best F-score: {1:0.5f}   Model: {2:}".format(
                    split_filename, fscore, log_file
                )
            )

        # Write average F-score for all splits to the results.txt file
        f_avg /= n_folds
        f.write(split_filename + ", " + str("avg") + ", " + str(f_avg) + "\n")
        f.flush()

    f.close()


if __name__ == "__main__":
    print_pkg_versions()

    parser = argparse.ArgumentParser("PyTorch implementation of VJMHT")
    parser.add_argument(
        "-d",
        "--datasets",
        type=str,
        help="Path to a comma separated list of h5 datasets",
    )
    parser.add_argument(
        "-s", "--splits", type=str, help="Comma separated list of split files."
    )
    parser.add_argument("-e", "--eval", action="store_true", help="eval")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Prints out more messages"
    )
    parser.add_argument("-c", "--config", type=str, default="cfg", help="config file")
    args = parser.parse_args()

    # MAIN
    # ======================
    args.config = args.config.replace("/", ".").replace(".py", "")
    config = importlib.import_module(args.config)
    EXP = args.config.split(".")[-1].upper()
    args.output_dir = os.path.join("results", EXP)
    os.makedirs(args.output_dir, exist_ok=True)
    hps = config.HParameters()
    engine = importlib.import_module("tools." + hps.engine)
    hps.load_from_args(args.__dict__)

    print("Parameters:")
    print("----------------------------------------------------------------------")
    print(hps)

    if not hps.eval:
        train(hps)

    results = [["No", "Split", "Mean F-score"]]

    start = time.time()
    for i, split_filename in enumerate(hps.splits):
        f_score = eval_split(hps, split_filename, data_dir=hps.output_dir)
        results.append([i + 1, split_filename, str(round(f_score * 100.0, 3)) + "%"])
    end = time.time()
    print(end - start)

    print("\nFinal Results:")
    print_table(results)
    sys.exit(0)
