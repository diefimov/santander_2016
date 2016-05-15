import glob
import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class RGFRegressor(BaseEstimator, RegressorMixin):

    rgf_parameters = ["reg_L2", "reg_sL2", "reg_depth", "algorithm", "loss",
                      "num_iteration_opt", "num_tree_search", "min_pop",
                      "opt_stepsize", "test_interval", "max_leaf_forest"]

    def __init__(self, sparse=False, reg_L2=0.001, reg_sL2=0.005,
                 reg_depth=1, algorithm="RGF", loss="LS", num_iteration_opt=7,
                 num_tree_search=10, min_pop=5, opt_stepsize=0.5,
                 test_interval=1000, max_leaf_forest=3000, verbose=True,
                 working_directory="/tmp/", rgf_bin="/usr/bin/rgf/"):

        self.sparse = sparse

        self.reg_L2 = reg_L2
        self.reg_sL2 = reg_sL2
        self.reg_depth = reg_depth
        self.algorithm = algorithm
        self.loss = loss
        self.num_iteration_opt = num_iteration_opt
        self.num_tree_search = num_tree_search
        self.min_pop = min_pop
        self.opt_stepsize = opt_stepsize
        self.test_interval = test_interval
        self.max_leaf_forest = max_leaf_forest
        self.verbose = verbose

        self.working_directory = working_directory
        self.rgf_bin = rgf_bin
        self.files = []

    def _get_random_filename(self, suffix):
        fname = tempfile.mktemp(suffix='_' + suffix, prefix='rgf',
                                dir=self.working_directory)
        self.files.append(fname)
        return fname

    def _write_config_file(self, mode="train", fname_x=None, fname_y=None,
                           fname_model=None, fname_predict=None):
        fname = self._get_random_filename(suffix="config_{}".format(mode))

        with open(fname + ".inp", "w") as out:
            if mode == "train":
                out.write("train_x_fn={}\n".format(fname_x))
                out.write("train_y_fn={}\n".format(fname_y))
                out.write("model_fn_prefix={}\n".format(fname_model))
                for param in self.rgf_parameters:
                    out.write("{}={}\n".format(param, getattr(self, param)))
                if self.verbose:
                    out.write("Verbose\n")
            elif mode == "predict":
                out.write("test_x_fn={}\n".format(fname_x))
                out.write("model_fn={}\n".format(fname_model))
                out.write("prediction_fn={}\n".format(fname_predict))
        return fname

    def _write_x_file(self, x):
        fname = self._get_random_filename(suffix="x")
        if self.sparse:
            # TODO
            assert NotImplemented
        else:
            pd.DataFrame(x).to_csv(fname, index=False, sep=' ', header=False)
        return fname

    def _write_y_file(self, y):
        fname = self._get_random_filename(suffix="y")
        pd.DataFrame(y).to_csv(fname, index=False, sep=' ', header=False)
        return fname

    def fit(self, x, y):
        fname_x = self._write_x_file(x)
        fname_y = self._write_y_file(y)
        self.fname_model = self._get_random_filename(suffix="model")
        fname_config = self._write_config_file(mode="train",
                                               fname_x=fname_x,
                                               fname_y=fname_y,
                                               fname_model=self.fname_model)
        fname_log = self._get_random_filename(suffix="log_train")

        # fit model
        os.system('perl {rgf_bin}/call_exe.pl {rgf_bin}/rgf train '
                  '{fname_config} >> {fname_log} 2>&1'.format(
                      rgf_bin=self.rgf_bin,
                      fname_config=fname_config,
                      fname_log=fname_log))

        return self

    def predict(self, x):
        return self.predict_many(x, return_last=True)

    def predict_many(self, x, return_last=False):
        fname_x = self._write_x_file(x)
        fname_log = self._get_random_filename(suffix="log_predict")

        preds = []

        for fname_model_ in sorted(glob.glob(self.fname_model + '*')):
            fname_predict = self._get_random_filename(suffix="predict")

            fname_config = self._write_config_file(mode="predict",
                                                   fname_x=fname_x,
                                                   fname_model=fname_model_,
                                                   fname_predict=fname_predict)

            # predict model
            os.system('perl {rgf_bin}/call_exe.pl {rgf_bin}/rgf predict '
                      '{fname_config} >> {fname_log} 2>&1'.format(
                          rgf_bin=self.rgf_bin,
                          fname_config=fname_config,
                          fname_log=fname_log))

            pred = np.loadtxt(fname_predict)
            preds.append((fname_model_, pred))

        if return_last:
            return preds[-1][1]
        else:
            return preds

    def clean_files(self):
        for fn in self.files:
            os.system("rm {}*".format(fn))
