# -*- coding: utf-8 -*-
"""
@author: Mathias Müller | Faron - kaggle.com/mmueller

https://github.com/Far0n/kaggletils
"""

import abc
from datetime import datetime

import numpy as np
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import log_loss


class CrossValidatorClfBase:
    __metaclass__ = abc.ABCMeta

    def __init__(self, params=None, seed=0):
        return

    @abc.abstractmethod
    def train(self, x_train, y_train, x_valid=None, y_valid=None, sample_weights=None):
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass


class CrossValidator(object):
    def __init__(self, clf, clf_params=None, nfolds=5, folds=None, stratified=True, seed=0, regression=False, nbags=1,
                 subsample=1., bootstrap=False, shuffle=False, metric=log_loss, average_oof=False, verbose=True):
        self.clf = clf
        self.clf_params = clf_params if clf_params is not None else {}
        self.nfolds = nfolds if folds is None else len(folds)
        self.stratified = None if folds is not None or nfolds < 1 else False if regression is True else stratified
        self.seed = seed
        self.regression = regression
        self.nbags = nbags
        self.subsample = subsample
        self.bootstrap = bootstrap
        self.shuffle = shuffle
        self.metric = metric
        self.verbose = verbose
        self.average_oof = average_oof if nfolds > 1 else False
        self.nclass = None
        self.pdim = None
        self.sample_weights = None
        self.subsets = False

        self.oof_train = None
        self.oof_test = None
        self.oof_probe = None
        self.elapsed_time = None
        self.cv_scores = None
        self.cv_mean = None
        self.cv_std = None
        self.folds = folds
        self.mean_train = None
        self.mean_test = None
        self.mean_probe = None

        self.train_score = None
        self.probe_score = None

        self.ntrain = None
        self.ntest = None
        self.nprobe = None

    def run_cv(self, x_train, y_train, x_test=None, x_probe=None, y_probe=None,
               sample_weights=None, subset_column=None, subset_values=None):
        ts = datetime.now()
        self.ntrain = x_train.shape[0]
        self.ntest = 0 if x_test is None else x_test.shape[0]
        self.nprobe = 0 if x_probe is None else x_probe.shape[0]
        self.sample_weights = sample_weights
        self.nclass = 1 if self.regression else np.unique(y_train).shape[0]
        self.pdim = 1 if self.nclass <= 2 else self.nclass

        if not isinstance(x_train, np.ndarray):
            subset_column = x_train.columns.get_loc(subset_column) if subset_column is not None else None
            x_train = np.array(x_train)
        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)
        if not isinstance(x_test, np.ndarray) and x_test is not None:
            x_test = np.array(x_test)
        if not isinstance(x_test, np.ndarray) and x_test is not None:
            x_probe = np.array(x_probe)
        if not isinstance(y_probe, np.ndarray) and y_probe is not None:
            y_probe = np.array(y_probe)
        if self.nclass <= 2:
            y_train = y_train.ravel()

        if self.verbose:
            if x_test is None and x_probe is None:
                print 'CrossValidator: X_train: {0}'.format(x_train.shape)
            elif x_test is not None and x_probe is None:
                print 'CrossValidator: X_train: {0}, X_test: {1}'.format(x_train.shape, x_test.shape)
            elif x_test is None and x_probe is not None:
                print 'CrossValidator: X_train: {0}, X_probe: {1}'.format(x_train.shape, x_probe.shape)
            else:
                print 'CrossValidator: X_train: {0}, X_test: {1}, X_probe: {2}'.\
                    format(x_train.shape, x_test.shape, x_probe.shape)

        if subset_column is None:
            self.__run_cv(x_train, y_train, x_test, x_probe, y_probe, sample_weights)
            return

        self.subsets = True
        oof_train = np.zeros((self.ntrain, self.pdim))
        oof_test = np.zeros((self.ntest, self.pdim))
        oof_probe = np.zeros((self.nprobe, self.pdim))

        if subset_values is None:
            subset_values = np.unique(x_train[:, subset_column])

        cv_scores = []

        if self.verbose:
            print '{0} Subsets (subset column: {1}, subset values: {2})' \
                .format(len(subset_values), subset_column, subset_values)

        for val in subset_values:
            train_ix = np.in1d(x_train[:, subset_column], val)
            test_ix = np.in1d(x_test[:, subset_column], val) if self.ntest > 0 else None
            probe_ix = np.in1d(x_probe[:, subset_column], val) if self.nprobe > 0 else None
            x_train_sub = x_train[train_ix]
            y_train_sub = y_train[train_ix]
            x_test_sub = x_test[test_ix] if self.ntest > 0 else None
            x_probe_sub = x_probe[probe_ix] if self.nprobe > 0 else None
            y_probe_sub = y_probe[probe_ix] if self.nprobe > 0 else None
            weights_sub = sample_weights[train_ix] if sample_weights is not None else None
            if self.verbose:
                if self.ntest > 0:
                    print 'Subset CV (column: {0}, value: {1}, ntrain: {2}, ntest: {3})' \
                        .format(subset_column, val, x_train_sub.shape[0], x_test_sub.shape[0])
                else:
                    print 'Subset CV (column: {0}, value: {1}, ntrain: {2})' \
                        .format(subset_column, val, x_train_sub.shape[0])
            self.__run_cv(x_train_sub, y_train_sub, x_test_sub, x_probe_sub, y_probe_sub, weights_sub)
            oof_train[train_ix] = self.oof_train
            oof_test[test_ix] = self.oof_test
            oof_probe[probe_ix] = self.oof_probe
            cv_scores.append(self.cv_scores)

        te = datetime.now()
        elapsed_time = (te - ts)

        self.oof_train = oof_train
        self.oof_test = oof_test
        self.oof_probe = oof_probe
        self.mean_train = np.mean(oof_train, axis=0)
        self.mean_test = np.mean(oof_test, axis=0) if self.ntest > 0 else None
        self.mean_probe = np.mean(oof_probe, axis=0) if self.nprobe > 0 else None
        self.cv_scores = cv_scores
        self.cv_mean = np.mean(cv_scores)
        self.cv_std = np.std(cv_scores)
        self.elapsed_time = elapsed_time
        self.train_score = self.metric(y_train, oof_train)
        self.probe_score = self.metric(y_probe, oof_probe) if self.nprobe > 0 else None

        if self.nclass <= 2:
            self.mean_train = self.mean_train[0]
            self.mean_test = self.mean_test[0] if self.ntest > 0 else None

        if self.verbose:
            print 'CV-Mean: {0:.12f}'.format(self.cv_mean)
            print 'CV-Std:  {0:.12f}'.format(self.cv_std)
            print 'Runtime: {0}'.format(elapsed_time)

    def __run_cv(self, x_train, y_train, x_test=None, x_probe=None, y_probe=None, sample_weights=None):
        ts = datetime.now()
        ntrain = x_train.shape[0]
        ntest = 0 if x_test is None else x_test.shape[0]
        nprobe = 0 if x_probe is None else x_probe.shape[0]
        prefix = '\t' if self.subsets else ''

        if self.folds is None:
            if self.nfolds > 1:
                if self.stratified:
                    self.folds = StratifiedKFold(y_train, n_folds=self.nfolds, shuffle=True, random_state=self.seed)
                else:
                    self.folds = KFold(self.ntrain, n_folds=self.nfolds, shuffle=True, random_state=self.seed)
            else:
                self.folds = [(np.arange(ntrain), [])]

        oof_train = np.zeros((ntrain, self.pdim))
        oof_test = np.zeros((ntest, self.pdim))
        oof_probe = np.zeros((nprobe, self.pdim))
        oof_test_folds = np.empty((self.nfolds, ntest, self.pdim))
        oof_probe_folds = np.empty((self.nfolds, nprobe, self.pdim))

        cv_scores = []

        if self.verbose:
            print prefix + '{0} Fold CV (seed: {1}, stratified: {2}, nbags: {3}, ' \
                           'subsample: {4}, bootstrap: {5}, shuffle: {6}, average oof: {7})' \
                .format(self.nfolds, self.seed, self.stratified, self.nbags,
                        self.subsample, self.bootstrap, self.shuffle, self.average_oof)

        if self.nfolds > 1:
            ts_cv = datetime.now()
            for i, (train_ix, valid_ix) in enumerate(self.folds):
                ts_fold = datetime.now()
                x_train_oof = x_train[train_ix]
                y_train_oof = y_train[train_ix]
                x_valid_oof = x_train[valid_ix]
                y_valid_oof = y_train[valid_ix]
                weights = sample_weights[train_ix] if sample_weights is not None else None

                if self.verbose:
                    print prefix + 'Fold {0:02d}: X_train: {1}, X_valid: {2}'. \
                        format(i + 1, x_train_oof.shape, x_valid_oof.shape)

                ntrain_oof = x_train_oof.shape[0]
                nvalid_oof = x_valid_oof.shape[0]

                oof_bag_valid = np.empty((self.nbags, nvalid_oof, self.pdim))
                oof_bag_test = np.empty((self.nbags, ntest, self.pdim))
                oof_bag_probe = np.empty((self.nbags, nprobe, self.pdim))

                for k in range(self.nbags):
                    ix = np.random.choice(ntrain_oof, int(self.subsample * ntrain_oof), self.bootstrap)
                    if not self.shuffle:
                        ix = np.sort(ix)
                    weights = sample_weights[ix] if sample_weights is not None else None
                    clf = self.clf(params=self.clf_params.copy(), seed=self.seed + k)
                    clf.train(x_train_oof[ix], y_train_oof[ix], x_valid_oof, y_valid_oof, weights)

                    oof_bag_valid[k, :, :] = clf.predict(x_valid_oof).reshape((-1, self.pdim))
                    if ntest > 0:
                        oof_bag_test[k, :, :] = clf.predict(x_test).reshape((-1, self.pdim))
                    if nprobe > 0:
                        oof_bag_probe[k, :, :] = clf.predict(x_probe).reshape((-1, self.pdim))

                pred_oof_valid = oof_bag_valid.mean(axis=0)
                oof_train[valid_ix, :] = pred_oof_valid
                if ntest > 0:
                    pred_oof_test = oof_bag_test.mean(axis=0)
                    oof_test_folds[i, :, :] = pred_oof_test
                if nprobe > 0:
                    pred_oof_probe = oof_bag_probe.mean(axis=0)
                    oof_probe_folds[i, :, :] = pred_oof_probe

                scr = self.metric(y_valid_oof, pred_oof_valid) if nvalid_oof > 0 else np.nan
                cv_scores.append(scr)

                scr_probe = self.metric(y_probe, pred_oof_probe) if nprobe > 0 else np.nan

                te_fold = datetime.now()
                if self.verbose:
                    print prefix + '         {0:.12f} ({1})'.format(scr, (te_fold - ts_fold))
                    print prefix + '         {0:.12f}'.format(scr_probe)
            te_cv = datetime.now()
        else:
            ts_cv = te_cv = datetime.now()
            cv_scores = np.nan

        self.cv_scores = cv_scores
        self.cv_mean = np.mean(cv_scores)
        self.cv_std = np.std(cv_scores)

        if ntest > 0 or nprobe > 0:
            if self.average_oof:
                if ntest > 0:
                    oof_test[:, :] = oof_test_folds.mean(axis=0)
                if nprobe > 0:
                    oof_probe[:, :] = oof_probe_folds.mean(axis=0)
            else:
                if self.verbose and self.nfolds > 0:
                    print prefix + 'CV-Mean: {0:.12f}'.format(self.cv_mean)
                    print prefix + 'CV-Std:  {0:.12f}'.format(self.cv_std)
                    print prefix + 'Runtime: {0}'.format((te_cv - ts_cv))
                if self.verbose:
                    print prefix + 'OnePass: X_train: {0}, X_test: {1}'. \
                        format(x_train.shape, x_test.shape)

                oof_bag_test = np.empty((self.nbags, ntest, self.pdim))
                oof_bag_probe = np.empty((self.nbags, nprobe, self.pdim))
                for k in range(self.nbags):
                    ix = np.random.choice(ntrain, int(self.subsample * ntrain), self.bootstrap)
                    if not self.shuffle:
                        ix = np.sort(ix)
                    clf = self.clf(params=self.clf_params.copy(), seed=self.seed + k)
                    weights = sample_weights[ix] if sample_weights is not None else None
                    clf.train(x_train[ix], y_train[ix], sample_weights=weights)
                    if ntest > 0:
                        oof_bag_test[k, :, :] = clf.predict(x_test).reshape((-1, self.pdim))
                    if nprobe > 0:
                        oof_bag_probe[k, :, :] = clf.predict(x_probe).reshape((-1, self.pdim))
                if ntest > 0:
                    oof_test[:, :] = oof_bag_test.mean(axis=0)
                if nprobe > 0:
                    oof_probe[:, :] = oof_bag_probe.mean(axis=0)

        te = datetime.now()
        elapsed_time = (te - ts)

        self.oof_train = oof_train
        self.oof_test = oof_test
        self.oof_probe = oof_probe
        self.mean_train = np.mean(oof_train, axis=0)
        self.mean_test = np.mean(oof_test, axis=0) if ntest > 0 else None
        self.mean_probe = np.mean(oof_probe, axis=0) if nprobe > 0 else None
        self.elapsed_time = elapsed_time
        self.train_score = self.metric(y_train, oof_train)
        self.probe_score = self.metric(y_probe, oof_probe) if self.nprobe > 0 else None

        if self.nclass <= 2:
            self.mean_train = self.mean_train[0]
            self.mean_test = self.mean_test[0] if ntest > 0 else None

        if self.verbose:
            if self.average_oof:
                print prefix + 'CV-Mean: {0:.12f}'.format(self.cv_mean)
                print prefix + 'CV-Std:  {0:.12f}'.format(self.cv_std)
            print prefix + 'Runtime: {0}'.format(elapsed_time)


    def print_cv_summary(self):
        if self.cv_scores is None:
            return
        for k in range(self.nfolds):
            print 'Fold {0:02d}: {1:.12f}'.format(k + 1, self.cv_scores[k])
        print 'CV-Mean: {0:.12f}'.format(self.cv_mean)
        print 'CV-Std:  {0:.12f}'.format(self.cv_std)
        print 'Runtime: {0}'.format(self.elapsed_time)

    @property
    def oof_predictions(self):
        return self.oof_train, self.oof_test

    @property
    def train_predictions(self):
        return self.oof_train

    @property
    def test_predictions(self):
        return self.oof_test

    @property
    def probe_predictions(self):
        return self.oof_probe

    @property
    def cv_stats(self):
        return self.cv_mean, self.cv_std

    @property
    def oof_means(self):
        return self.mean_train, self.mean_test

    @property
    def oof_means_delta(self):
        return np.abs(self.mean_train - self.mean_test)
