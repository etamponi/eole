import unittest

import numpy

from core import matrix_utils
from core.interfaces import Weigher


__author__ = 'Emanuele'


class MatrixUtilsTest(unittest.TestCase):

    def test_order(self):
        ordering_matrix = numpy.asarray([
            [5, 6, 4],
            [2, 1, 3]
        ])
        expected = [
            [1, 0, 2],
            [2, 0, 1]
        ]
        self.assertEqual(expected, matrix_utils.order(ordering_matrix).tolist())

    def test_sort_matrix(self):
        matrix = numpy.asarray([
            [1, 3, 2],
            [5, 6, 4]
        ])
        indices = numpy.asarray([
            [0, 2, 1],
            [2, 0, 1]
        ])
        sorted_matrix = matrix_utils.sort_matrix(matrix, indices).tolist()
        self.assertEqual([[1, 2, 3], [4, 5, 6]], sorted_matrix)
        self.assertIsNot(matrix, sorted_matrix)

    def test_partial_row_average_matrix(self):
        a = numpy.array([0.7, 0.3])
        b = numpy.array([0.1, 0.9])
        c = numpy.array([0.6, 0.4])
        d = 1 - a
        e = 1 - c
        f = 1 - b
        matrix = numpy.asarray([
            [a, b, c],
            [d, e, f]
        ])
        expected = numpy.asarray([
            [a, (a + b)/2, (a + b + c)/3],
            [d, (d + e)/2, (d + e + f)/3]
        ])

        pram = matrix_utils.partial_row_average_matrix(matrix)
        numpy.testing.assert_array_almost_equal(expected, pram)
        self.assertIsNot(matrix, pram)

        weights_matrix = numpy.asarray([
            [3, 2, 1],
            [1, 2, 3]
        ])
        weights_matrix_bak = weights_matrix.copy()
        expected = numpy.asarray([
            [a, (3*a + 2*b)/5, (3*a + 2*b + c)/6],
            [d, (d + 2*e)/3, (d + 2*e + 3*f)/6]
        ])

        pram = matrix_utils.partial_row_average_matrix(matrix, weights_matrix)
        numpy.testing.assert_array_almost_equal(expected, pram)
        self.assertIsNot(matrix, pram)
        numpy.testing.assert_array_equal(weights_matrix_bak, weights_matrix)

    def test_sharpen_probability_matrix(self):
        a = numpy.asarray([0.7, 0.3])
        b = numpy.asarray([0.1, 0.9])
        c = numpy.asarray([0.6, 0.4])
        d = 1 - a
        e = 1 - c
        f = 1 - b
        matrix = numpy.asarray([
            [a, b, c],
            [d, e, f]
        ])
        g = numpy.asarray([1., 0.])
        h = numpy.asarray([0., 1.])
        expected = numpy.asarray([
            [g, h, g],
            [h, h, g]
        ])
        sm = matrix_utils.sharpen_probability_matrix(matrix)
        numpy.testing.assert_array_equal(expected, sm)
        self.assertIsNot(matrix, sm)

    def test_prediction_matrix(self):
        a = numpy.asarray([0.7, 0.3])
        b = numpy.asarray([0.1, 0.9])
        c = numpy.asarray([0.6, 0.4])
        d = 1 - a
        e = 1 - c
        f = 1 - b
        matrix = numpy.asarray([
            [a, b, c],
            [d, e, f]
        ])
        labels = numpy.asarray(["a", "b"])
        expected = numpy.asarray([
            ["a", "b", "a"],
            ["b", "b", "a"]
        ])
        numpy.testing.assert_array_equal(expected, matrix_utils.prediction_matrix(matrix, labels))

    def test_competence_matrix(self):
        weigher = FakeWeigher()
        instances = numpy.zeros((5, 3))
        instances[:, 0] = numpy.asarray([3.0, -1., 2.0, 1.0, 5.0])
        centroids = numpy.asarray([
            [1.5, 0.0, 0.0],
            [-2., 0.0, 0.0]
        ])
        expected_matrix = numpy.asarray([
            [4.5, -6.0],
            [-1.5, 2.0],
            [3.0, -4.0],
            [1.5, -2.0],
            [7.5, -10.]
        ])
        numpy.testing.assert_array_equal(expected_matrix, matrix_utils.competence_matrix(instances, centroids, weigher))

    def test_probability_matrix(self):
        instances = numpy.asarray([
            [0.1, 0.4, 0.6],
            [0.9, 1.1, 0.9],
            [0.2, 0.3, 1.7],
            [0.5, 1.5, 1.5]
        ])
        experts = [FakeExpert(start=0), FakeExpert(start=1)]
        n_labels = 2
        expected_matrix = numpy.asarray([
            [[0.2, 0.8], [0.4, 0.6]],
            [[.45, .55], [.55, .45]],
            [[0.4, 0.6], [.15, .85]],
            [[.25, .75], [0.5, 0.5]],
        ])
        numpy.testing.assert_array_almost_equal(
            expected_matrix, matrix_utils.probability_matrix(instances, experts, n_labels)
        )


class FakeWeigher(Weigher):
    def get_weights(self, instances, centroid):
        # For testing purposes. Returns centroid[0] * x[0] as weight for each instance x
        return numpy.asarray([centroid[0] * x[0] for x in instances])

    def train(self, instances):
        pass


class FakeExpert(object):
    def __init__(self, start):
        self.start = start

    def predict_probs(self, instances):
        # For testing purposes. Uses features (start, start+1) as probabilities (after normalization)
        return numpy.asarray([self._fake_prob(x) for x in instances])

    def _fake_prob(self, x):
        p = x[[self.start, self.start+1]]
        return p / p.sum()
