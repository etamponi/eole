import unittest

import numpy
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.tree.tree import DecisionTreeClassifier

from analysis.dataset_utils import ArffLoader
from analysis.experiment import Experiment
from core.centroid_picker import RandomCentroidPicker
from core.ensemble_trainer import EnsembleTrainer
from core.eole import EOLE
from core.exponential_weigher import ExponentialWeigher


__author__ = 'Emanuele Tamponi'


class ExperimentTest(unittest.TestCase):

    def setUp(self):
        self.ensemble = EOLE(
            n_experts=2,
            ensemble_trainer=EnsembleTrainer(
                DecisionTreeClassifier(max_depth=1),
                centroid_picker=RandomCentroidPicker(),
                sampling=ExponentialWeigher(precision=1, power=2)
            ),
            expert_weighting=ExponentialWeigher(precision=1, power=2),
            preprocessor=MinMaxScaler(),
            use_probs=True,
            use_competences=False
        )
        self.experiment = Experiment(
            "test_experiment", self.ensemble, ArffLoader("tests/dataset.arff"), folds=2, repetitions=10
        )

    def test_run_returns_report(self):
        report = self.experiment.run()

        self.assertEqual(20, report.sample_size)

    def test_each_repetition_is_different(self):
        report = self.experiment.run()

        self.assertFalse(numpy.array_equal(report.accuracy_sample[0], report.accuracy_sample[10]))