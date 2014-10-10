import unittest

import numpy
from scipy.spatial import distance

from core import matrix_utils
import mri


__author__ = 'Emanuele Tamponi'


class MRITest(unittest.TestCase):

    def test_probe(self):
        instances = numpy.asarray([
            [0., 0.],
            [1., 0.],
            [0., 3.],
            [0., 4.],
            [1., 3.]
        ])
        distances = distance.squareform(distance.pdist(instances, metric="euclidean"))
        indices = matrix_utils.order(distances, decreasing=False)
        labels = numpy.asarray(["a", "a", "b", "b", "a"])
        expected_values = numpy.asarray([
            [2, 0],
            [2, 0],
            [1, 2],
            [0, 2],
            [1, 1]
        ], dtype=numpy.float64)
        numpy.testing.assert_array_equal(expected_values, mri.probe(distances, indices, labels, radius=1.3))

    def test_psi(self):
        labels = numpy.asarray(["a", "a", "b", "b", "a"])
        probes = numpy.asarray([
            [2, 1],
            [3, 0],
            [1, 2],
            [1, 2],
            [1, 2]
        ], dtype=numpy.float64)
        expected_psi = (1. / 3.) * numpy.asarray([
            (2 - 1),
            (3 - 0),
            (2 - 1),
            (2 - 1),
            (1 - 2)
        ])
        numpy.testing.assert_array_almost_equal(expected_psi, mri.psi(probes, labels))

    def test_psi_non_binary(self):
        labels = numpy.asarray(["a", "b", "c"])
        probes = numpy.asarray([
            [1, 2, 2],
            [2, 2, 2],
            [1, 2, 3]
        ]).astype(numpy.float64)
        expected_psi = numpy.asarray([
            (1 - 4./2) / 5,
            (2 - 4./2) / 6,
            (3 - 3./2) / 6
        ])
        numpy.testing.assert_array_almost_equal(expected_psi, mri.psi(probes, labels))

    def test_mri(self):
        psi_matrix = numpy.asarray([
            [0.8, -0.2],
            [-0.1, 0.3],
            [0.3, 0.2]
        ])
        expected_mri = (1. / (2 * (1 + 0.5))) * numpy.asarray([
            (1 - 0.8) + (1 - -0.2)*0.5,
            (1 - -0.1) + (1 - 0.3)*0.5,
            (1 - 0.3) + (1 - 0.2)*0.5
        ])
        numpy.testing.assert_array_almost_equal(expected_mri, mri.mri(psi_matrix))

    def test_find_radius_for_percent(self):
        distances = numpy.asarray([
            [3, 1, 2],
            [2, 6, 4],
            [1, 1.5, 0.5]
        ])
        indices = matrix_utils.order(distances, decreasing=False)
        expected_radius = float(2 + 4 + 1) / 3
        self.assertAlmostEqual(expected_radius, mri.mean_radius_for_percent(distances, indices, 70))
        expected_radius = float(3 + 6 + 1.5) / 3
        self.assertAlmostEqual(expected_radius, mri.mean_radius_for_percent(distances, indices, 100))
