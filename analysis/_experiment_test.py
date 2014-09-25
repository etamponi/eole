import unittest

from sklearn.preprocessing.data import MinMaxScaler
from sklearn.tree.tree import DecisionTreeClassifier

from analysis import Experiment
from analysis.arff_utils import ArffLoader
from eole import EOLE
from eole.centroid_picker import RandomCentroidPicker
from eole.ensemble_trainer import EnsembleTrainer
from eole.exponential_weighting import ExponentialWeighting


__author__ = 'Emanuele Tamponi'


class ExperimentTest(unittest.TestCase):

    def test_run_returns_report(self):
        ensemble = EOLE(
            n_experts=2,
            ensemble_trainer=EnsembleTrainer(
                DecisionTreeClassifier(max_depth=1),
                centroid_picker=RandomCentroidPicker(),
                sampling=ExponentialWeighting(precision=1, power=2)
            ),
            expert_weighting=ExponentialWeighting(precision=1, power=2),
            preprocessor=MinMaxScaler(),
            use_probs=True,
            use_competences=False
        )
        exp = Experiment("test_experiment", ensemble, ArffLoader("tests/dataset.arff"), folds=2, repetitions=10)
        report = exp.run()
        self.assertEqual(20, report.sample_size)
