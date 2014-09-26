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

            if self.label_attribute is None:
                self.label_attribute = dataset["attributes"][-1][0]

            data = list(numpy.asarray(dataset["data"]).transpose())
            labels = None

            row = 0
            for attribute_name, attribute_type in dataset["attributes"]:
                if attribute_name == self.label_attribute:
                    # Labels found!
                    labels = data.pop(row)
                    continue
                # Nominal attribute
                if isinstance(attribute_type, list):
                    # Check if no data is present for this attribute
                    if all(e is None for e in data[row]):
                        # If no data is present, just remove the row
                        data.pop(row)
                        continue
                    data[row] = numpy.asarray(label_binarize(data[row], attribute_type), dtype=numpy.float64)
                else:
                    # Numeric attributes: reshape to do hstack
                    data[row] = data[row].reshape((len(data[row]), 1)).astype(numpy.float64)
                # Check next row if we have not removed the current one
                row += 1

            instances = numpy.hstack(tuple(data))

            return instances, labels
