import unittest

import numpy

from experiment import arff_utils
from experiment.arff_utils import ArffLoader


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
        self.assertEqual([1, 3], arff_utils.nominal_attributes(dataset))

    def test_load_dataset_standard(self):
        al = ArffLoader("tests/dataset.arff")
        expected_instances = numpy.asarray([
            [0.0, 0.0, 1.0, 1.2],
            [1.0, 0.0, 0.0, 0.3],
            [0.0, 1.0, 0.0, 3.0]
        ])
        expected_labels = numpy.asarray([
            "POS",
            "POS",
            "NEG"
        ])
        instances, labels = al.load_dataset()
        numpy.testing.assert_array_equal(expected_instances, instances)
        numpy.testing.assert_array_equal(expected_labels, labels)

    def test_load_dataset_with_label_not_last_attribute(self):
        al = ArffLoader("tests/dataset.arff", label_attribute="first")
        expected_instances = numpy.asarray([
            [0.0, 1.0, 1.2],
            [0.0, 1.0, 0.3],
            [1.0, 0.0, 3.0]
        ])
        expected_labels = numpy.asarray([
            "sheep",
            "cannon",
            "egg"
        ])
        instances, labels = al.load_dataset()
        numpy.testing.assert_array_equal(expected_instances, instances)
        numpy.testing.assert_array_equal(expected_labels, labels)