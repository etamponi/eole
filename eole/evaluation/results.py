from itertools import product
import matplotlib
import numpy
import os
from scipy.stats.stats import ttest_ind
import cPickle
from eole.analysis.report import Report
from eole.evaluation import dataset_names

__author__ = 'Emanuele Tamponi'


CLASSIFIERS = ["boosting_logreg", "bagging_logreg", "random_forest"]
LEGEND = {
    "boosting_logreg": "AdaBoost",
    "bagging_logreg": "Bagging",
    "random_forest": "Random Forest"
}

CANDIDATE_PERCENTS = [0.3]  # [0.1, 0.3, 0.5]
LEAF_NUMS = [50]  # [50, 100, None]
SAMPLE_PERCENTS = [100]  # [50, 100]

NL = "\n"


def main():
    configure_matplotlib()
    from matplotlib import pyplot

    for candidate_percent, leaf_num, sample_percent in product(CANDIDATE_PERCENTS, LEAF_NUMS, SAMPLE_PERCENTS):
        get_table_data(get_eole_name(candidate_percent, leaf_num, sample_percent))
        comparison_table(candidate_percent, leaf_num, sample_percent)
        scatter_accuracies(pyplot, candidate_percent, leaf_num, sample_percent)

    scatter_best_accuracies(pyplot)

    synthesis_table()


def synthesis_table():
    all_info = {}
    for i, (candidate_percent, leaf_num, sample_percent) in enumerate(product(CANDIDATE_PERCENTS, LEAF_NUMS,
                                                                              SAMPLE_PERCENTS)):
        eole_name = get_eole_name(candidate_percent, leaf_num, sample_percent)
        all_info[eole_name] = get_table_data(eole_name)["info_sum"]

    table_name = "synthesis_flt"
    with open("figures/{}.tex".format(table_name), "w") as f:
        f.writelines((r"\begin{table}\centering", NL))
        f.writelines((r"\renewcommand{\arraystretch}{1.2}", NL))
        # f.writelines((r"\renewcommand{\tabcolsep}{0pt}", NL))
        f.writelines((r"\small", NL))
        f.writelines((r"\begin{tabularx}{0.70\textwidth}{XXXr@{/}r@{/}lr@{/}r@{/}lr@{/}r@{/}l}", NL))
        f.writelines((r"\toprule", NL))
        f.writelines((
            r"\multicolumn{3}{c}{FLT parameters} & \multicolumn{3}{c}{} & ",
            r"\multicolumn{3}{c}{} & \multicolumn{3}{c}{Random} \\", NL,
            r"\cmidrule(lr){1-3}", NL,
            r"\candidatenum & \samplepercent & \maxleaves & ",
            r"\multicolumn{3}{c}{AdaBoost} & \multicolumn{3}{c}{Bagging} & \multicolumn{3}{c}{Forest} \\", NL
        ))
        f.writelines((r"\midrule", NL))
        for candidate_percent in CANDIDATE_PERCENTS:
            write_candidate = True
            for sample_percent in SAMPLE_PERCENTS:
                write_sample = True
                for leaf_num in LEAF_NUMS:
                    eole_name = get_eole_name(candidate_percent, leaf_num, sample_percent)
                    if write_candidate:
                        f.write(r"\multirow{6}{*}{%d\%%} " % int(100*candidate_percent))
                        write_candidate = False
                    f.write(r"& ")
                    if write_sample:
                        f.write(r"\multirow{3}{*}{%d\%%} " % sample_percent)
                        write_sample = False
                    f.write(r"& ")
                    f.write(r"{}".format(leaf_num if leaf_num is not None else r"$\infty$"))
                    for classifier in CLASSIFIERS:
                        win, tie, loss = tuple(all_info[eole_name][classifier])
                        f.write(r"& {} & {} & {} ".format(win, tie, loss))
                    f.writelines((r"\\", NL))
                if sample_percent == 50:
                    f.writelines((r"\cmidrule{2-12}", NL))
            if candidate_percent != 0.5:
                f.writelines((r"\midrule", NL))
        f.writelines((
            r"\bottomrule", NL,
            r"\end{tabularx}", NL,
            r"\caption{Overview of the results: win/tie/loss triplets.}", NL,
            r"\label{tab:%s}" % table_name, NL,
            r"\end{table}", NL
        ))


def scatter_accuracies(pyplot, candidate_percent, leaf_num, sample_percent):
    eole_accuracies = numpy.zeros(len(dataset_names()))
    cls_accuracies = numpy.zeros_like(eole_accuracies)
    eole_name = get_eole_name(candidate_percent, leaf_num, sample_percent)
    table_data = get_table_data(eole_name)
    for i, dataset in enumerate(dataset_names()):
        eole_acc = table_data[dataset]["eole"][0]
        if eole_acc > eole_accuracies[i]:
            eole_accuracies[i] = eole_acc
        for classifier in CLASSIFIERS:
            cls_acc = table_data[dataset][classifier][0]
            if cls_acc > cls_accuracies[i]:
                cls_accuracies[i] = cls_acc

    pyplot.plot(numpy.linspace(70, 100), numpy.linspace(70, 100), "k--")
    pyplot.scatter(100*cls_accuracies, 100*eole_accuracies, c="k")

    pyplot.xlim(70, 100)
    pyplot.ylim(70, 100)
    pyplot.xticks(numpy.linspace(70, 100, 7))
    pyplot.yticks(numpy.linspace(70, 100, 7))
    pyplot.xlabel("Best other")
    pyplot.ylabel("FLT")
    pyplot.axes().set_aspect(1)
    pyplot.grid()

    pyplot.gcf().set_size_inches(3, 3)
    pyplot.savefig("figures/plot_comparison_{}.pdf".format(eole_name), bbox_inches="tight")
    pyplot.close()


def scatter_best_accuracies(pyplot):
    eole_accuracies = numpy.zeros(len(dataset_names()))
    cls_accuracies = numpy.zeros_like(eole_accuracies)
    for candidate_percent, leaf_num, sample_percent in product(CANDIDATE_PERCENTS, LEAF_NUMS, SAMPLE_PERCENTS):
        eole_name = get_eole_name(candidate_percent, leaf_num, sample_percent)
        table_data = get_table_data(eole_name)
        for i, dataset in enumerate(dataset_names()):
            eole_acc = table_data[dataset]["eole"][0]
            if eole_acc > eole_accuracies[i]:
                eole_accuracies[i] = eole_acc
            # Only the first time
            if cls_accuracies[i].sum() == 0:
                for classifier in CLASSIFIERS:
                    cls_acc = table_data[dataset][classifier][0]
                    if cls_acc > cls_accuracies[i]:
                        cls_accuracies[i] = cls_acc

    pyplot.plot(numpy.linspace(70, 100), numpy.linspace(70, 100), "k--")
    pyplot.scatter(100*cls_accuracies, 100*eole_accuracies, c="k", s=12)

    pyplot.xlim(70, 100)
    pyplot.ylim(70, 100)
    pyplot.xticks(numpy.linspace(70, 100, 7))
    pyplot.yticks(numpy.linspace(70, 100, 7))
    pyplot.xlabel("Best other")
    pyplot.ylabel("Best FLT")
    pyplot.axes().set_aspect(1)
    pyplot.grid()

    pyplot.gcf().set_size_inches(3, 3)
    pyplot.savefig("figures/plot_comparison_overall.pdf", bbox_inches="tight")
    pyplot.close()


def comparison_table(candidate_percent, leaf_num, sample_percent):
    eole_name = get_eole_name(candidate_percent, leaf_num, sample_percent)
    table_name = "comparison_vs_logreg_{}".format(eole_name)

    table_data = get_table_data(eole_name)
    with open("figures/{}.tex".format(table_name), "w") as f:
        f.writelines((r"\begin{table}\centering", NL, r"\label{tab:%s}" % table_name, NL))
        f.writelines((r"\renewcommand{\arraystretch}{1.1}", NL))
        f.writelines((r"\renewcommand{\tabcolsep}{0pt}", NL))
        f.writelines((r"\small", NL))
        f.writelines((r"\begin{tabularx}{0.95\textwidth}{Xr@{$\pm$}llr@{$\pm$}llr@{$\pm$}llr@{$\pm$}ll}", NL))
        f.writelines((r"\toprule", NL))
        f.writelines((
            r" & \twocol{} & & \twocol{} & & \twocol{} & & \twocol{Random} & \\", NL,
            r"Dataset & \twocol{FLT} & \phantom{abc} & \twocol{AdaBoost} & \phantom{abc} & ",
            r"\twocol{Bagging} & \phantom{abc} & \twocol{Forest} & \phantom{abc} \\", NL,
            r"\midrule", NL
        ))
        for dataset in dataset_names():
            f.write(r"{} & ".format(dataset))
            eole_mean, eole_std = table_data[dataset]["eole"]
            f.write(r"${:.2f}$ & ${:.1f}$ & ".format(100*eole_mean, 100*eole_std))
            for classifier in CLASSIFIERS:
                cls_mean, cls_std, info = table_data[dataset][classifier]
                f.write(r"& ${:.2f}$ & ${:.1f}$ & ".format(100*cls_mean, 100*cls_std))
                if info[0] == 1:    # Classifier wins
                    f.write(r"$\circ$ ")
                elif info[2] == 1:  # FLT wins
                    f.write(r"$\bullet$ ")
            f.writelines((r"\\", NL))
        f.writelines((r"\midrule", NL))
        f.write(r"(Win/Tie/Loss) & \multicolumn{3}{c}{} ")
        info_sum = table_data["info_sum"]
        for classifier in CLASSIFIERS:
            win, tie, loss = tuple(info_sum[classifier])
            f.write(r"& \multicolumn{2}{c}{(%d/%d/%d)} & " % (win, tie, loss))
        f.writelines((
            r"\\", NL,
            r"\bottomrule", NL,
            r"\end{tabularx}", NL
        ))
        caption = (
            r"Comparison for \flt with \candidatenum = {}\%, \maxleaves = {}, and \samplepercent = {}\%.".format(
            int(100*candidate_percent), leaf_num if leaf_num is not None else r"$\infty$", sample_percent)
        )
        f.writelines((r"\caption{%s}" % caption, NL))
        f.writelines((r"\end{table}", NL))


def get_table_data(eole_name):
    data_file = "reports/small/comparison_vs_logreg_{}.dat".format(eole_name)
    if os.path.isfile(data_file):
        with open(data_file) as f:
            return cPickle.load(f)
    table_data = {}
    info_sum = {
        "boosting_logreg": numpy.zeros(3, dtype=float),
        "bagging_logreg": numpy.zeros(3, dtype=float),
        "random_forest": numpy.zeros(3, dtype=float)
    }
    for dataset in dataset_names():
        row = {}
        table_data[dataset] = row
        eole_results = Report.load("reports/small/{}_{}.rep".format(dataset, eole_name))
        eole_accuracy = eole_results.accuracy_sample[:, -1]
        row["eole"] = (eole_accuracy.mean(), eole_accuracy.std())
        for classifier in CLASSIFIERS:
            cls_results = Report.load("reports/small/{}_small_{}.rep".format(dataset, classifier))
            cls_accuracy = cls_results.accuracy_sample[:, -1]
            value, p = ttest_ind(eole_accuracy, cls_accuracy)
            info = numpy.zeros(3)
            if p <= 0.05:
                if value < 0:
                    info[0] = 1
                else:
                    info[2] = 1
            else:
                info[1] = 1
            row[classifier] = (cls_accuracy.mean(), cls_accuracy.std(), info)
            info_sum[classifier] += info
    table_data["info_sum"] = info_sum
    with open(data_file, "w") as f:
        cPickle.dump(table_data, f)
    return table_data


def get_eole_name(candidate_percent, leaf_num, sample_percent):
    return "small_eole_{:02d}_{}{}".format(
        int(100*candidate_percent),
        "050" if leaf_num == 50 else ("100" if leaf_num == 100 else "Nil"),
        "_50" if sample_percent == 50 else ""
    )


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


if __name__ == "__main__":
    main()
