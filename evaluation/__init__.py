import glob
import os
import re

__author__ = 'Emanuele Tamponi'


def dataset_names(start=0, n=None):
    dataset_dir = os.path.join(os.path.dirname(__file__), "datasets")
    names = []
    for dataset_path in glob.glob("{}/*.arff".format(dataset_dir)):
        names.append(re.search(r"([\w\-]+)\.arff", dataset_path).group(1))
    if start >= len(names):
        return []
    elif n is None or start + n > len(names):
        return names[start:]
    else:
        return names[start:start+n]


if __name__ == "__main__":
    print dataset_names()
    print dataset_names(1, 3)
    print dataset_names(1, 50)