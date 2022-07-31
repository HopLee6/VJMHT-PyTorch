from torch.autograd import Variable


class HParameters:
    def __init__(self):

        # models
        self.engine = "engine"
        self.model_name = "reccotrans"
        self.nhead1 = 2
        self.nhid1 = 4096
        self.nlayers1 = 2
        self.shot_dim = 512
        self.dropout1 = 0

        self.nhead2 = 2
        self.nhid2 = 2048
        self.nlayers2 = 3
        self.dropout2 = 0

        # training
        self.l2_req = 0.000001
        self.lr_epochs = [0, 30]
        self.lr = [0.00001, 0.000001]
        self.epochs_max = 60
        self.regular = True
        self.eval_freq = 1
        self.rec_factor = 1
        self.supervised_factor = 1

        # data
        self.datasets = [
            "datasets/eccv16_dataset_summe_google_pool5.h5",
            "datasets/eccv16_dataset_tvsum_google_pool5.h5",
            "datasets/eccv16_dataset_ovp_google_pool5.h5",
            "datasets/eccv16_dataset_youtube_google_pool5.h5",
        ]

        # self.splits = ['splits/summe_splits.json',
        #                 'splits/tvsum_splits.json']
        self.splits = ["splits/summe_trans_splits.json"]

        # general
        self.rnd_seed = 0
        self.verbose = False
        self.use_cuda = True
        self.cuda_device = 0
        self.max_summary_length = 0.15
        self.rand_choice = False

        return

    def get_dataset_by_name(self, dataset_name):
        for d in self.datasets:
            if dataset_name in d:
                return [d]
        return None

    def load_from_args(self, args):
        for key in args:
            val = args[key]
            if val is not None:
                if hasattr(self, key) and isinstance(getattr(self, key), list):
                    val = val.split()

                setattr(self, key, val)

    def __str__(self):
        vars = [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr))
            and not (attr.startswith("__") or attr.startswith("_"))
        ]

        info_str = ""
        for i, var in enumerate(vars):
            val = getattr(self, var)
            if isinstance(val, Variable):
                val = val.data.cpu().numpy().tolist()[0]
            info_str += "[" + str(i) + "] " + var + ": " + str(val) + "\n"

        return info_str
