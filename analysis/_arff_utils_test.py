import unittest

import numpy

from analysis import arff_utils
from analysis.arff_utils import ArffLoader


__author__ = 'Emanuele Tamponi'


class ArffUtilsTest(unittest.TestCase):

    def test_identify_nominal(self):
        dataset = {
            'description': 'Example Dataset',
            'relation': 'Example',
            'attributes': [
                ('input1', 'REAL'),
                ('input2', ["sheep", "egg", "farm"]),
                ('input3', "REAL"),
                ('input4', ["queen", "king", "jack"]),
                ('y', 'REAL'),
            ],
            'data': [
                [0.0, "sheep", 0.0, "jack", 0.0],
                [0.0, "egg", 1.0, "king", 1.0]
            ]
        }
        expected = {
            "input2": {"index": 1, "classes": ["sheep", "egg", "farm"]},
            "input4": {"index": 3, "classes": ["queen", "king", "jack"]}
        }
        self.assertEqual(expected, arff_utils.nominal_attributes(dataset))

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
        instances, labels = al.load_dataset()
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
        instances, labels = al.load_dataset()
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
        instances, _ = al.load_dataset()
        numpy.testing.assert_array_equal(expected_instances, instances)

    def test_load_dataset_with_unknown_feature(self):
        al = ArffLoader("tests/dataset_with_unknown_feature.arff")
        expected_instances = numpy.asarray([
            [1.2],
            [0.3],
            [3.0],
            [4.0]
        ])
        instances, _ = al.load_dataset()
        numpy.testing.assert_array_equal(expected_instances, instances)