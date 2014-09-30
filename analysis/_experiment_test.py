import unittest

import numpy

from analysis.experiment import Experiment


__author__ = 'Emanuele Tamponi'


class ExperimentTest(unittest.TestCase):

    def setUp(self):
        self.experiment = Experiment(
            name="test_experiment",
            ensemble=FakeEOLE(n_experts=5),
            dataset_loader=FakeLoader(n=100, n_features=5, n_labels=3),
            folds=2, repetitions=10
        )

    def test_run_returns_report(self):
        report = self.experiment.run()
        self.assertEqual(20, report.sample_size)

    def test_each_repetition_is_different(self):
        report = self.experiment.run()
        self.assertFalse(numpy.array_equal(report.accuracy_sample[0], report.accuracy_sample[10]))


class FakeEOLE(object):

    def __init__(self, n_experts):
        self.n_experts = n_experts
        self.labels = None

    def train(self, instances, labels):
        self.labels = numpy.unique(labels)

    def predict(self, instances):
        return numpy.asarray(
            [numpy.random.choice(self.labels, size=len(instances)) for _ in range(self.n_experts)]
        ).transpose()


class FakeLoader(object):

    def __init__(self, n, n_features, n_labels):
        self.n = n
        self.n_features = n_features
        self.n_labels = n_labels

    def load_dataset(self):
        return numpy.random.randn(self.n, self.n_features), numpy.random.choice(self.n_labels, size=self.n)