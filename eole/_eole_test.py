import unittest

import numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from eole import EOLE
from eole.centroid_picker import RandomCentroidPicker
from eole.ensemble_trainer import EnsembleTrainer
from eole.exponential_weigher import ExponentialWeigher


__author__ = 'Emanuele Tamponi'


class EOLETest(unittest.TestCase):

    def test_eole_trains(self):
        ensemble = EOLE(
            n_experts=5,
            ensemble_trainer=EnsembleTrainer(
                DecisionTreeClassifier(max_depth=1, max_features="auto"),
                centroid_picker=RandomCentroidPicker(),
                sampling=ExponentialWeigher(precision=1, power=2)
            ),
            expert_weighting=ExponentialWeigher(precision=1, power=2),
            preprocessor=MinMaxScaler(),
            use_probs=False,
            use_competences=False
        )
        instances = numpy.random.randn(50, 10)
        labels = numpy.random.choice(["a", "b"], size=50)
        ensemble.train(instances, labels)
        self.assertEqual(5, len(ensemble.experts))

    def test_eole_works_without_preprocessor(self):
        ensemble = EOLE(
            n_experts=5,
            ensemble_trainer=EnsembleTrainer(
                DecisionTreeClassifier(max_depth=1, max_features="auto"),
                centroid_picker=RandomCentroidPicker(),
                sampling=ExponentialWeigher(precision=1, power=2)
            ),
            expert_weighting=ExponentialWeigher(precision=1, power=2),
            preprocessor=None,
            use_probs=False,
            use_competences=False
        )
        instances = numpy.random.randn(50, 10)
        labels = numpy.random.choice(["a", "b"], size=50)
        ensemble.train(instances, labels)
        self.assertEqual(5, len(ensemble.experts))
