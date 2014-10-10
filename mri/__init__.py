import numpy


__author__ = 'Emanuele Tamponi'


def probe(distances, indices, labels, radius):
    classes = numpy.unique(labels).tolist()
    ret = numpy.zeros((len(distances), len(classes)))
    for i, col_indices in enumerate(indices):
        for col_index in col_indices:
            if distances[i, col_index] > radius:
                break
            ret[i, classes.index(labels[col_index])] += 1
    return ret


def psi(probes, labels):
    classes = numpy.unique(labels).tolist()
    ret = numpy.zeros(len(probes))
    for i, (p, l) in enumerate(zip(probes, labels)):
        same = p[classes.index(l)]
        ret[i] = (same - (p.sum() - same)/(len(classes) - 1)) / p.sum()
    return ret


def psi_matrix(distances, indices, labels, radii):
    matrix = numpy.zeros((len(distances), len(radii)))
    for j, radius in enumerate(radii):
        probes = probe(distances, indices, labels, radius)
        matrix[:, j] = psi(probes, labels)
    return matrix


def mri(psi_m):
    m = psi_m.shape[1]
    mris = numpy.zeros(len(psi_m))
    ws = numpy.linspace(1, 1. / m, num=m)
    for j, w in enumerate(ws):
        mris += w * (1 - psi_m[:, j])
    return mris / (2 * ws.sum())


def mean_radius_for_percent(distances, indices, percent):
    j_percent = max(int(float(percent) * distances.shape[1] / 100) - 1, 0)
    dist_col = numpy.asarray([distances[row, indices[row, j_percent]] for row in range(len(distances))])
    return dist_col.mean()
