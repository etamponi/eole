import numpy

__author__ = 'Emanuele'


def order(ordering_matrix):
    return numpy.fliplr(numpy.argsort(ordering_matrix))


def sort_matrix(matrix, indices):
    matrix = matrix.copy()
    for i in range(len(matrix)):
        matrix[i] = matrix[i][indices[i]]
    return matrix


def partial_row_average_matrix(matrix, weights_matrix=None):
    assert len(matrix.shape) == 3
    if weights_matrix is None:
        weights_matrix = numpy.ones(matrix.shape[:-1])
    else:
        weights_matrix = weights_matrix.copy()
    pram = matrix.copy()
    for j in range(matrix.shape[1]):
        # Multiply each column by the corresponding weight column
        pram[:, j, :] = numpy.dot(numpy.diag(weights_matrix[:, j]), pram[:, j])
        if j > 0:
            # Accumulate columns (sum part of the average)
            pram[:, j, :] += pram[:, j - 1, :]
            # Accumulate weights (to divide the average)
            weights_matrix[:, j] += weights_matrix[:, j - 1]
    # "Invert" weights matrix, that now contains the partial sums of the weights
    weights_matrix = 1. / weights_matrix
    for j in range(matrix.shape[1]):
        # This is like dividing the weighted sum by the sum of the weights
        pram[:, j, :] = numpy.dot(numpy.diag(weights_matrix[:, j]), pram[:, j])
    return pram


def sharpen_probability_matrix(matrix):
    sm = numpy.zeros_like(matrix)
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            sm[i][j][v.argmax()] = 1.0
    return sm


def prediction_matrix(probability_matrix, labels):
    pm = []
    for row in probability_matrix:
        pm.append(labels[row.argmax(axis=1)])
    return numpy.asarray(pm)
