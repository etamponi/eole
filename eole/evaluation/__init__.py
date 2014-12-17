import glob
import os
import re

__author__ = 'Emanuele Tamponi'


def dataset_names(n_groups=1, group=0):
    dataset_dir = os.path.join(os.path.dirname(__file__), "datasets")
    names = []
    for dataset_path in glob.glob("{}/*.arff".format(dataset_dir)):
        names.append(re.search(r"([\w\-]+)\.arff", dataset_path).group(1))
    group_size = float(len(names))/n_groups
    return names[int(group*group_size):int((group+1)*group_size)]


if __name__ == "__main__":
    print dataset_names()
    print dataset_names(4, 0)
    print dataset_names(4, 1)
    print dataset_names(4, 2)
    print dataset_names(4, 3)
