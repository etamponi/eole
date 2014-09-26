from cmath import sqrt
import unittest

import numpy

from eole.ensemble_gate import EnsembleGate
from eole.local_expert import LocalExpert
from eole.mocks import InverseDistance, FakeEstimator


__author__ = 'Emanuele'


class EnsembleGateTest(unittest.TestCase):

    def setUp(self):
        self.gate = EnsembleGate(
            experts=[
                LocalExpert(FakeEstimator(n_labels=2, label_cycle=[0, 1], prob=0.7), None),
                LocalExpert(FakeEstimator(n_labels=2, label_cycle=[1, 1], prob=0.9), None),
                LocalExpert(FakeEstimator(n_labels=2, label_cycle=[1, 0], prob=0.6), None)
            ],
            expert_weighting=InverseDistance(1, 1)
        )
        self.gate.experts[0].centroid = numpy.asarray([0, 0])
        self.gate.experts[1].centroid = numpy.asarray([-1, 0])
        self.gate.experts[2].centroid = numpy.asarray([1, 0])
        self.instances = numpy.asarray([
            [-1, -1],
            [1, 1]
        ])

    def test_competence_matrix(self):
        expected = [
            [1./(sqrt(2).real+1), 1./(1+1), 1./(sqrt(5).real+1)],
            [1./(sqrt(2).real+1), 1./(sqrt(5).real+1), 1./(1+1)]
        ]
        self.assertEqual(expected, self.gate.competence_matrix(self.instances).tolist())

    def test_probability_matrix(self):
        expected = [
            [[0.7, 1.0 - 0.7], [1.0 - 0.9, 0.9], [0.4, 0.6]],
            [[1.0 - 0.7, 0.7], [1.0 - 0.9, 0.9], [0.6, 0.4]]
        ]
        self.assertEqual(expected, self.gate.probability_matrix(self.instances).tolist())
