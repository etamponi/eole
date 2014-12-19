import glob
import multiprocessing
import os
import re
import signal

__author__ = 'Emanuele Tamponi'


def dataset_names(n_groups=1, group=0):
    dataset_dir = os.path.join(os.path.dirname(__file__), "datasets")
    names = []
    for dataset_path in glob.glob("{}/*.arff".format(dataset_dir)):
        names.append(re.search(r"([\w\-]+)\.arff", dataset_path).group(1))
    group_size = float(len(names))/n_groups
    return names[int(group*group_size):int((group+1)*group_size)]


def run_parallel(function, argument_list, processes=3):
    pool = multiprocessing.Pool(processes=processes, initializer=init_worker)
    try:
        for arguments in argument_list:
            pool.apply_async(function, args=arguments)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print "Keyboard Interrupt, terminating..."
        pool.terminate()
        pool.join()


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


if __name__ == "__main__":
    print dataset_names()
    print dataset_names(4, 0)
    print dataset_names(4, 1)
    print dataset_names(4, 2)
    print dataset_names(4, 3)
