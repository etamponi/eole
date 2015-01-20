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

N = 50
S = 50
K = 7

inputs = numpy.random.rand(N, 2)

centroid_indices = numpy.random.choice(N, size=K)

for index in centroid_indices:
    pyplot.figure(figsize=(3, 3))
    pyplot.grid()
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)
    pyplot.xticks(numpy.linspace(0, 1, 6), ["" for _ in range(6)])
    pyplot.yticks(numpy.linspace(0, 1, 6), ["" for _ in range(6)])
    pyplot.axes().set_aspect(1)

    centroid = inputs[index]
    for i, x in enumerate(inputs):
        weight = math.exp(-0.5 * 7 * distance.euclidean(x, centroid))
        pyplot.scatter([x[0]], [x[1]], c="k", s=S*weight, marker="s" if i == index else "o",
                       alpha=1.0 if i == index else 0.5)

    pyplot.savefig("weighted_dataset_{:02d}.pdf".format(index), bbox_inches="tight")
    pyplot.close()
