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

    def test_run_returns_results(self):
        ensemble = EOLE(
            ensemble_trainer=EnsembleTrainer(
                DecisionTreeClassifier(max_depth=1),
                n_experts=2,
                centroid_picker=RandomCentroidPicker(),
                sampling=ExponentialWeighting(precision=1, power=2)
            ),
            expert_weighting=ExponentialWeighting(precision=1, power=2),
            preprocessor=MinMaxScaler(),
            use_probs=True,
            use_competences=False
        )
        exp = Experiment("test_experiment", ensemble, ArffLoader("tests/dataset.arff"), folds=2, repetitions=10)
        result = exp.run()
        self.assertEqual(20, len(result.runs))
        self.assertEqual((2, 2), result.runs[0]["prediction_matrix"].shape)
        self.assertEqual(2, len(result.runs[0]["labels"]))
        self.assertEqual(exp, result.experiment)
