import numpy

__author__ = 'Emanuele'


class CentroidPicker(object):

    def __init__(self):
        pass

    def pick(self, instances, n_centroids):
        assert n_centroids <= len(instances)
        choices = numpy.random.choice(len(instances), size=n_centroids, replace=False)
        return numpy.asarray([instances[i] for i in choices])