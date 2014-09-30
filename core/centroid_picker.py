import numpy
from sklearn.neighbors.kde import KernelDensity


__author__ = 'Emanuele'


class RandomCentroidPicker(object):

    def __init__(self):
        pass

    def pick(self, instances, n_centroids):
        assert n_centroids <= len(instances)
        choices = numpy.random.choice(len(instances), size=n_centroids, replace=False)
        return instances[choices]


class AlmostRandomCentroidPicker(object):

    def __init__(self):
        pass

    def pick(self, instances, n_centroids):
        p = numpy.ones(len(instances)) / len(instances)
        ret = [instances[numpy.random.choice(len(instances), p=p)]]
        while len(ret) < n_centroids:
            distances = numpy.asarray([numpy.linalg.norm(x - ret[-1]) for x in instances])
            p = p * numpy.log(1.0 + distances)
            p = p / p.sum()
            ret.append(instances[numpy.random.choice(len(instances), p=p)])
        return numpy.asarray(ret)


class KernelDensityCentroidPicker(object):

    def __init__(self):
        pass

    def pick(self, instances, n_centroids):
        estimator = KernelDensity().fit(instances)
        p = numpy.exp(estimator.score_samples(instances))
        p = p / p.sum()
        ret = [instances[numpy.random.choice(len(instances), p=p)]]
        while len(ret) < n_centroids:
            distances = numpy.asarray([numpy.linalg.norm(x - ret[-1]) for x in instances])
            p = p * numpy.log(1.0 + distances)
            p = p / p.sum()
            ret.append(instances[numpy.random.choice(len(instances), p=p)])
        return numpy.asarray(ret)


class DeterministicCentroidPicker(object):

    def __init__(self):
        pass

    def pick(self, instances, n_centroids):
        estimator = KernelDensity().fit(instances)
        p = numpy.exp(estimator.score_samples(instances))
        p = p / p.sum()
        ret = [instances[p.argmax()]]
        while len(ret) < n_centroids:
            distances = numpy.asarray([numpy.linalg.norm(x - ret[-1]) for x in instances])
            p = p * numpy.log(1.0 + distances)
            p = p / p.sum()
            ret.append(instances[p.argmax()])
        return numpy.asarray(ret)
