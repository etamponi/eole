from copy import deepcopy

import numpy
from sklearn.cross_validation import StratifiedKFold

from analysis.report import Report


__author__ = 'Emanuele Tamponi'


class Experiment(object):

    def __init__(self, name, ensemble, dataset_loader, folds, repetitions):
        self.name = name
        self.ensemble = ensemble
        self.dataset_loader = dataset_loader
        self.folds = folds
        self.repetitions = repetitions

    def run(self):
        report = Report(self)
        instances, labels = self.dataset_loader.load_dataset()
        for i in range(self.repetitions):
            instances, labels = instances.copy(), labels.copy()
            numpy.random.seed(i)
            permutation = numpy.random.permutation(len(instances))
            instances = instances[permutation]
            labels = labels[permutation]
            cv = StratifiedKFold(labels, n_folds=self.folds)
            for train_indices, test_indices in cv:
                train_instances = instances[train_indices]
                train_labels = labels[train_indices]
                test_instances = instances[test_indices]
                test_labels = labels[test_indices]
                ensemble = deepcopy(self.ensemble)
                ensemble.train(train_instances, train_labels)
                report.analyze_run(ensemble.predict(test_instances), test_labels)
        return report