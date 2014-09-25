import arff
import numpy
from sklearn.preprocessing.data import OneHotEncoder
from sklearn.preprocessing.label import LabelEncoder

__author__ = 'Emanuele Tamponi'


def nominal_attributes(dataset):
    ret = []
    for i, (_, attribute_type) in enumerate(dataset["attributes"]):
        if isinstance(attribute_type, list):
            ret.append(i)
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
            label_index = [i for i, a in enumerate(dataset["attributes"]) if a[0] == self.label_attribute][0]
            instances = numpy.asarray(dataset["data"])
            labels = instances[:, label_index]
            nominal_indices = nominal_attributes(dataset)
            nominal_indices.remove(label_index)
            real_instances = numpy.zeros_like(instances, dtype=float)
            lab_enc = LabelEncoder()
            for j in range(instances.shape[1]):
                if j in nominal_indices:
                    lab_enc.fit(dataset["attributes"][j][1])
                    real_instances[:, j] = lab_enc.transform(instances[:, j])
                elif j != label_index:
                    real_instances[:, j] = instances[:, j]
            real_instances = numpy.delete(real_instances, label_index, axis=1)
            nominal_indices = [i if i < label_index else i - 1 for i in nominal_indices]
            real_instances = OneHotEncoder(categorical_features=nominal_indices).fit_transform(real_instances).toarray()
            return real_instances, labels
