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
                    # Convert None in '?' for next check and to make label_binarize work
                    for j in range(len(data[row])):
                        if data[row][j] is None:
                            data[row][j] = "?"
                    if numpy.all(data[row] == "?"):
                        # If no data is present, just remove the row
                        data.pop(row)
                        continue
                    data[row] = numpy.asarray(label_binarize(data[row], attribute_type), dtype=numpy.float64)
                else:
                    # Numeric attributes: check for nan values
                    data[row] = data[row].astype(numpy.float64)
                    nans = numpy.isnan(data[row])
                    if numpy.all(nans):
                        # If everything is nan, remove the feature
                        data.pop(row)
                        continue
                    if numpy.any(nans):
                        mean = data[row][numpy.invert(nans)].sum() / numpy.invert(nans).sum()
                        data[row][nans] = mean
                    # Reshape to do hstack later
                    data[row] = data[row].reshape((len(data[row]), 1))
                # Check next row if we have not removed the current one
                row += 1

            instances = numpy.hstack(tuple(data))

            return instances, labels
