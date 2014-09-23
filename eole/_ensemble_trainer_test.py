import unittest

import numpy
from sklearn.tree.tree import DecisionTreeClassifier

from eole.centroid_picker import CentroidPicker
from eole.ensemble_trainer import EnsembleTrainer
from eole.exponential_weighting import ExponentialWeighting


__author__ = 'Emanuele'


class EnsembleTrainerTest(unittest.TestCase):

    def test_train(self):
        instances, labels = numpy.random.randn(50, 10), numpy.random.choice(["a", "b", "c"], size=50)
        tests = numpy.random.randn(10, 10)
        ft = EnsembleTrainer(
            base_estimator=DecisionTreeClassifier(),
            n_experts=2, centroid_picker=CentroidPicker(), sampling=ExponentialWeighting(1, 2)
        )
        e = ft.train(instances, labels)
        self.assertEqual(2, len(e))
        self.assertNotEqual(e[0].centroid.tolist(), e[1].centroid.tolist())
        self.assertNotEqual(e[0].predict(tests).tolist(), e[1].predict(tests).tolist())
