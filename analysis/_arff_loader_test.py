import unittest

import numpy

from analysis.dataset_utils import ArffLoader


__author__ = 'Emanuele Tamponi'


class ArffLoaderTest(unittest.TestCase):

    def test_load_dataset_standard(self):
        al = ArffLoader("tests/dataset.arff")
        expected_instances = numpy.asarray([
            [1.0, 0.0, 0.0, 1.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 4.0]
        ])
        expected_labels = numpy.asarray([
            "POS",
            "POS",
            "NEG",
            "NEG"
        ])
        instances, labels = al.get_dataset()
        numpy.testing.assert_array_equal(expected_instances, instances)
        numpy.testing.assert_array_equal(expected_labels, labels)

    def test_load_dataset_with_label_not_last_attribute(self):
        al = ArffLoader("tests/dataset.arff", label_attribute="first")
        expected_instances = numpy.asarray([
            [1.2, 0.0],
            [0.3, 0.0],
            [3.0, 1.0],
            [4.0, 1.0]
        ])
        expected_labels = numpy.asarray([
            "sheep",
            "cannon",
            "egg",
            "cannon"
        ])
        instances, labels = al.get_dataset()
        numpy.testing.assert_array_equal(expected_instances, instances)
        numpy.testing.assert_array_equal(expected_labels, labels)

    def test_load_dataset_with_unknowns_nominals(self):
        al = ArffLoader("tests/dataset_with_unknowns.arff")
        expected_instances = numpy.asarray([
            [1.0, 0.0, 0.0, 1.2],
            [0.0, 0.0, 0.0, 0.3],
            [0.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 4.0]
        ])
        instances, _ = al.get_dataset()
        numpy.testing.assert_array_equal(expected_instances, instances)

    def test_load_dataset_with_unknown_feature(self):
        al = ArffLoader("tests/dataset_with_unknown_feature.arff")
        expected_instances = numpy.asarray([
            [1.2],
            [0.3],
            [3.0],
            [4.0]
        ])
        instances, _ = al.get_dataset()
        numpy.testing.assert_array_equal(expected_instances, instances)

    def test_load_dataset_with_unknown_real_values(self):
        al = ArffLoader("tests/dataset_with_unknown_real_values.arff")
        expected_instances = numpy.asarray([
            [1.2],
            [(1.2+3.6)/2],
            [3.6],
            [3.0],
            [4.0],
            [(3.0+4.0)/2]
        ])
        instances, _ = al.get_dataset()
        numpy.testing.assert_array_almost_equal(expected_instances, instances)

    def test_load_dataset_with_useless_real_values(self):
        al = ArffLoader("tests/dataset_with_useless_real_values.arff")
        expected_instances = numpy.asarray([
            [1.2],
            [0.3],
            [3.0],
            [4.0]
        ])
        instances, _ = al.get_dataset()
        numpy.testing.assert_array_almost_equal(expected_instances, instances)

    def test_load_dataset_with_useless_categorical_values(self):
        al = ArffLoader("tests/dataset_with_useless_categorical_values.arff")
        expected_instances = numpy.asarray([
            [1.2],
            [0.3],
            [3.0],
            [4.0]
        ])
        instances, _ = al.get_dataset()
        numpy.testing.assert_array_almost_equal(expected_instances, instances)