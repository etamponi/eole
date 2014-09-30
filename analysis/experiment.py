from copy import deepcopy

from sklearn.cross_validation import StratifiedKFold

from analysis.dataset_utils import CoupledShuffle
from analysis.report import Report


__author__ = 'Emanuele Tamponi'


class Experiment(object):

    def __init__(self, name, ensemble, dataset_loader, n_folds, n_repetitions):
        self.name = name
        self.ensemble = ensemble
        self.dataset_loader = dataset_loader
        self.n_folds = n_folds
        self.n_repetitions = n_repetitions

    def run(self):
        report = Report(self)
        shuffler = CoupledShuffle(*self.dataset_loader.load_dataset())
        for i in range(self.n_repetitions):
            instances, labels = shuffler.shuffle(seed=i)
            cv = StratifiedKFold(labels, n_folds=self.n_folds)
            for j, (train_indices, test_indices) in enumerate(cv):
                print "Experiment {}: start repetition {}, fold {}".format(self.name, i+1, j+1)
                ensemble = deepcopy(self.ensemble)
                # Training
                train_instances = instances[train_indices]
                train_labels = labels[train_indices]
                ensemble.train(train_instances, train_labels)
                # Testing
                test_instances = instances[test_indices]
                test_labels = labels[test_indices]
                report.analyze_run(ensemble.predict(test_instances), test_labels)
        return report
