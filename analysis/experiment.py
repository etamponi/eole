from copy import deepcopy

from sklearn.cross_validation import StratifiedKFold

from analysis.dataset_utils import CoupledShuffle
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
        shuffler = CoupledShuffle(*self.dataset_loader.load_dataset())
        for i in range(self.repetitions):
            instances, labels = shuffler.shuffle(seed=i)
            cv = StratifiedKFold(labels, n_folds=self.folds)
            for j, (train_indices, test_indices) in enumerate(cv):
                print "Start experiment {}: repetition {}, fold {}".format(self.name, i+1, j+1)
                train_instances = instances[train_indices]
                train_labels = labels[train_indices]
                test_instances = instances[test_indices]
                test_labels = labels[test_indices]
                ensemble = deepcopy(self.ensemble)
                ensemble.train(train_instances, train_labels)
                report.analyze_run(ensemble.predict(test_instances), test_labels)
        return report