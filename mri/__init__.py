import numpy
from scipy.spatial import distance

from core import matrix_utils


__author__ = 'Emanuele Tamponi'


def probe(instances, labels, radius, metric="euclidean"):
    classes = numpy.unique(labels).tolist()
    distances = distance.squareform(distance.pdist(instances, metric=metric))
    indices = matrix_utils.order(distances, decreasing=False)
    ret = numpy.zeros((len(instances), len(classes)))
    for i, row in enumerate(indices):
        for index in row:
            if distances[i, index] > radius:
                break
            ret[i, classes.index(labels[index])] += 1
    return ret


def psi(probes, labels):
    classes = numpy.unique(labels).tolist()
    ret = numpy.zeros(len(probes))
    for i, (p, l) in enumerate(zip(probes, labels)):
        same = p[classes.index(l)]
        ret[i] = (same - (p.sum() - same)/(len(classes) - 1)) / p.sum()
    return ret


def psi_matrix(instances, labels, radii, metric="euclidean"):
    matrix = numpy.zeros((len(instances), len(radii)))
    for j, radius in enumerate(radii):
        probes = probe(instances, labels, radius, metric)
        matrix[:, j] = psi(probes, labels)
    return matrix


def mri(psi_m):
    m = psi_m.shape[1]
    mris = numpy.zeros(len(psi_m))
    ws = numpy.linspace(1, 1. / m, num=m)
    for j, w in enumerate(ws):
        mris += w * (1 - psi_m[:, j])
    return mris / (2 * ws.sum())
