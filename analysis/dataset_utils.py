import arff
import numpy
from sklearn.preprocessing.label import label_binarize, LabelEncoder

__author__ = 'Emanuele Tamponi'


class ArffLoader(object):

    def __init__(self, file_name, label_attribute=None, binarize=True):
        self.file_name = file_name
        self.label_attribute = label_attribute
        self.binarize = binarize
        self.data, self.labels = self._load_dataset()

    def _load_dataset(self):
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
                if self.binarize:
                    data[row] = numpy.asarray(label_binarize(data[row], attribute_type), dtype=numpy.float64)
                else:
                    encoder = LabelEncoder()
                    encoder.classes_ = attribute_type
                    if "?" not in encoder.classes_:
                        encoder.classes_.insert(0, "?")
                    data[row] = encoder.transform(data[row]).reshape((len(data[row]), 1)).astype(numpy.float64)
            else:
                # Numeric attributes: check for nan values
                data[row] = data[row].astype(numpy.float64)
                nans = numpy.isnan(data[row])
                if numpy.all(nans):
                    # If everything is nan, remove the feature
                    data.pop(row)
                    continue
                # Reshape to do hstack later
                data[row] = data[row].reshape((len(data[row]), 1))
            # Check if feature contains information
            if numpy.all(data[row].var(axis=0) == 0):
                data.pop(row)
                continue
            # Go to next row only if we have NOT removed the current one
            row += 1

        for label in numpy.unique(labels):
            label_mask = labels == label
            for feature_matrix in data:
                for i, feature in enumerate(feature_matrix.T):
                    label_feature = feature[label_mask]
                    nans = numpy.isnan(label_feature)
                    if numpy.any(nans):
                        if numpy.invert(nans).sum() == 0:
                            mean = 0.0
                        else:
                            mean = label_feature[numpy.invert(nans)].sum() / numpy.invert(nans).sum()
                        label_feature[nans] = mean
                        feature_matrix[labels == label, i] = label_feature
        return data, labels

    def get_dataset(self, feature_indices=None):
        if feature_indices is None:
            instances = numpy.hstack(tuple(self.data))
        else:
            instances = numpy.hstack(tuple(self.data[i] for i in feature_indices))
        return instances.copy(), self.labels

    def feature_num(self):
        return len(self.data)


class CoupledShuffle(object):

    def __init__(self, *arrays):
        self.arrays = arrays
        self.size = len(arrays[0])

    def shuffle(self, seed=None):
        if seed is not None:
            numpy.random.seed(seed)
        permutation = numpy.random.permutation(self.size)
        arrays = [array.copy()[permutation] for array in self.arrays]
        return arrays
