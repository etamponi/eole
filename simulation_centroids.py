import math
import matplotlib
import numpy
from scipy.spatial import distance

__author__ = 'Emanuele Tamponi'


def configure_matplotlib():
    matplotlib.use('pgf')
    pgf_rc = {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,     # use inline math for ticks
        "pgf.rcfonts": False,    # don't setup fonts from rc parameters
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": [
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage{microtype}",
            r"\usepackage{amsfonts}",
            r"\usepackage{amsmath}",
            r"\usepackage{amssymb}",
            r"\usepackage{booktabs}",
            r"\usepackage{fancyhdr}",
            r"\usepackage{graphicx}",
            r"\usepackage{nicefrac}",
            r"\usepackage{xspace}"
        ]
    }
    matplotlib.rcParams.update(pgf_rc)


configure_matplotlib()
from matplotlib import pyplot
numpy.random.seed(3)

N = 30
S = 30
K = 3

inputs = numpy.random.rand(N, 2)

pyplot.figure(figsize=(3, 3))
pyplot.grid()
pyplot.xlim(0, 1)
pyplot.ylim(0, 1)
pyplot.xticks(numpy.linspace(0, 1, 6), ["" for _ in range(6)])
pyplot.yticks(numpy.linspace(0, 1, 6), ["" for _ in range(6)])
pyplot.axes().set_aspect(1)

pyplot.scatter(inputs[:, 0], inputs[:, 1], c="k", alpha=0.5, s=S)
pyplot.savefig("situation_initial.pdf", bbox_inches="tight")
pyplot.close()

probabilities = 1.0 / N * numpy.ones(N)

centroid_indices = []
while len(centroid_indices) < K:
    pyplot.figure(figsize=(3, 3))
    pyplot.grid()
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)
    pyplot.xticks(numpy.linspace(0, 1, 6), ["" for _ in range(6)])
    pyplot.yticks(numpy.linspace(0, 1, 6), ["" for _ in range(6)])
    pyplot.axes().set_aspect(1)

    curr_index = numpy.random.choice(N, p=probabilities)
    centroid_indices.append(curr_index)

    for i, x in enumerate(inputs):
        probabilities[i] *= math.log(1 + distance.euclidean(x, inputs[curr_index]))
    probabilities /= probabilities.sum()

    for i, (x, y, p) in enumerate(zip(inputs[:, 0], inputs[:, 1], probabilities)):
        if i not in centroid_indices:
            pyplot.scatter([x], [y], c="k", alpha=0.5, s=S*N*p)
    for index in centroid_indices:
        pyplot.scatter([inputs[index, 0]], [inputs[index, 1]], c="k", marker="s", s=2*S)
    pyplot.savefig("situation_after_{}.pdf".format(len(centroid_indices)), bbox_inches="tight")
    pyplot.close()
