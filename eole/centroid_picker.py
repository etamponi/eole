from cmath import log
import numpy
from scipy.spatial import distance

__author__ = 'Emanuele'


class CentroidPicker(object):

    def __init__(self):
        pass

    def pick(self, instances, n_centroids):
        assert n_centroids <= len(instances)
        choices = numpy.random.choice(len(instances), size=n_centroids, replace=False)
        return numpy.asarray([instances[i] for i in choices])


class AlmostRandomCentroidPicker(object):

    def __init__(self):
        pass

    def pick(self, instances, n_centroids):
        p = numpy.ones(len(instances)) / len(instances)
        ret = [instances[numpy.random.choice(len(instances), p=p)]]
        while len(ret) < n_centroids:
            for i, x in enumerate(instances):
                p[i] *= log(1.0 + distance.euclidean(x, ret[-1])).real
            p = p / p.sum()
            ret.append(instances[numpy.random.choice(len(instances), p=p)])
        return numpy.asarray(ret)
