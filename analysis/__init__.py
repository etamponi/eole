from copy import deepcopy

import numpy
from sklearn.cross_validation import StratifiedKFold


__author__ = 'Emanuele Tamponi'


class Result(object):

    def __init__(self, experiment):
        self.experiment = experiment
        self.runs = []


class Experiment(object):

    def __init__(self, name, ensemble, dataset_loader, folds, repetitions):
        self.name = name
        self.ensemble = ensemble
        self.dataset_loader = dataset_loader
        self.folds = folds
        self.repetitions = repetitions

    def run(self):
        result = Result(self)
        instances, labels = self.dataset_loader.load_dataset()
        for i in range(self.repetitions):
            numpy.random.seed(i)
            # TODO shuffle instances and labels
            cv = StratifiedKFold(labels, n_folds=self.folds)
            for train_indices, test_indices in cv:
                train_instances = instances[train_indices]
                train_labels = labels[train_indices]
                test_instances = instances[test_indices]
                test_labels = labels[test_indices]
                ensemble = deepcopy(self.ensemble)
                ensemble.train(train_instances, train_labels)
                result.runs.append({"prediction_matrix": ensemble.predict(test_instances), "labels": test_labels})
        return result