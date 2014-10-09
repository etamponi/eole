import numpy
from scipy.spatial import distance
from sklearn.neighbors.kde import KernelDensity

import mri


__author__ = 'Emanuele'


class RandomCentroidPicker(object):

    def __init__(self):
        pass

    def pick(self, instances, labels, n_centroids):
        choices = numpy.random.choice(len(instances), size=n_centroids, replace=True)
        return instances[choices]


class AlmostRandomCentroidPicker(object):

    def __init__(self, dist_measure=distance.euclidean):
        self.dist_measure = dist_measure

    def pick(self, instances, labels, n_centroids):
        p = numpy.ones(len(instances)) / len(instances)
        centroids = numpy.zeros((n_centroids, instances.shape[1]))
        centroids[0] = instances[numpy.random.multinomial(1, p).argmax()]
        for i in range(1, n_centroids):
            distances = numpy.asarray([self.dist_measure(x, centroids[i-1]) for x in instances])
            p = p * numpy.log(1.0 + distances)
            p = p / p.sum()
            centroids[i] = instances[numpy.random.multinomial(1, p).argmax()]
        return centroids


class KernelDensityCentroidPicker(object):

    def __init__(self, bandwidth=1.0, dist_measure=distance.euclidean):
        self.bandwidth = bandwidth
        self.dist_measure = dist_measure

    def pick(self, instances, labels, n_centroids):
        estimator = KernelDensity(bandwidth=self.bandwidth).fit(instances)
        p = numpy.exp(estimator.score_samples(instances))
        p = p / p.sum()
        centroids = numpy.zeros((n_centroids, instances.shape[1]))
        centroids[0] = instances[numpy.random.multinomial(1, p).argmax()]
        for i in range(1, n_centroids):
            distances = numpy.asarray([self.dist_measure(x, centroids[i-1]) for x in instances])
            p = p * numpy.log(1.0 + distances)
            p = p / p.sum()
            centroids[i] = instances[numpy.random.multinomial(1, p).argmax()]
        return centroids


class DeterministicCentroidPicker(object):

    def __init__(self):
        pass

    def pick(self, instances, labels, n_centroids):
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


class MRIPicker(object):

    def __init__(self, m=10, max_radius=1):
        self.m = m
        self.max_radius = float(max_radius)

    def pick(self, instances, labels, n_centroids):
        radii = numpy.linspace(self.max_radius / self.m, self.max_radius, num=self.m)
        mris = mri.mri(mri.psi_matrix(instances, labels, radii, "chebyshev"))
        # mris = mris / mris.max()
        mris = numpy.abs(mris - 0.5)
        indices = mris.argsort()
        centroids = numpy.zeros((n_centroids, instances.shape[1]))
        for i in range(n_centroids):
            centroids[i] = instances[indices[i/20]]
        return centroids
