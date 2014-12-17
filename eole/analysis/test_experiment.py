import unittest

import numpy

from eole.analysis.experiment import Experiment


__author__ = 'Emanuele Tamponi'


class ExperimentTest(unittest.TestCase):

    def setUp(self):
        self.experiment = Experiment(
            name="test_experiment",
            ensemble=FakeEOLE(n_experts=5),
            dataset_loader=FakeLoader(n=100, n_features=5, n_labels=3),
            n_folds=2, n_repetitions=10
        )

    def test_run_returns_report(self):
        report = self.experiment.run()
        self.assertEqual(20, report.sample_size)

    def test_each_repetition_is_different(self):
        report = self.experiment.run()
        self.assertFalse(numpy.array_equal(report.accuracy_sample[0], report.accuracy_sample[2]))

    def test_each_run_is_equal(self):
        report1 = self.experiment.run()
        report2 = self.experiment.run()
        numpy.testing.assert_array_equal(report1.accuracy_sample, report2.accuracy_sample)


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

    def get_dataset(self):
        numpy.random.seed(1)
        return numpy.random.randn(self.n, self.n_features), numpy.random.choice(self.n_labels, size=self.n)