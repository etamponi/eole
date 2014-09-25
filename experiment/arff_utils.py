import arff
import numpy
from sklearn.preprocessing.label import label_binarize

__author__ = 'Emanuele Tamponi'


def nominal_attributes(dataset):
    ret = {}
    for i, (attribute_name, attribute_type) in enumerate(dataset["attributes"]):
        if isinstance(attribute_type, list):
            ret[attribute_name] = {"index": i, "classes": attribute_type}
    return ret


class ArffLoader(object):

    def __init__(self, file_name, label_attribute=None):
        self.file_name = file_name
        self.label_attribute = label_attribute

    def load_dataset(self):
        with open(self.file_name) as f:
            dataset = arff.load(f)

            nominals = nominal_attributes(dataset)
            if self.label_attribute is None:
                self.label_attribute = dataset["attributes"][-1][0]
            label_index = nominals[self.label_attribute]["index"]
            del nominals[self.label_attribute]

            data = list(numpy.asarray(dataset["data"]).transpose())
            labels = data[label_index]

            for nominal, info in nominals.iteritems():
                j, classes = info["index"], info["classes"]
                data[j] = numpy.asarray(label_binarize(data[j], classes))
            data.pop(label_index)

            for j in range(len(data)):
                if len(data[j].shape) < 2:
                    data[j] = data[j].reshape((len(data[j]), 1))
            instances = numpy.hstack(tuple(data)).astype(numpy.float64)

            return instances, labels
