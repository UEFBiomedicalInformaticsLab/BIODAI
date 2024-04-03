import os

from plots.archives.automated_hofs_archive import flatten_hofs_for_dataset_cv
from plots.barplot import barplot_with_std_to_file, barplot_to_file
from plots.plot_labels import ALL_CV_DATASETS, ALL_MAIN_NO_NSGA3
from saved_solutions.run_measure.run_best_dice import RunBestDice
from saved_solutions.run_measure.run_cross_hypervolume import RunCrossHypervolume
from saved_solutions.run_measure.run_fold_jaccard import RunFoldJaccard
from saved_solutions.run_measure.run_soft_cross_hypervolume import RunSoftCrossHypervolume
from saved_solutions.run_measure.run_test_hypervolume import RunTestHypervolume
from saved_solutions.run_measure.run_weight_overlap import RunWeightOverlap
from saved_solutions.solutions_from_files import solutions_from_files

SUMMARY_STAT_DIR = "summary_stats"
MAIN_LABS = ALL_MAIN_NO_NSGA3

if __name__ == '__main__':
    global_measures = [RunWeightOverlap(), RunBestDice()]
    fold_measures = [RunTestHypervolume(), RunCrossHypervolume(), RunSoftCrossHypervolume(), RunFoldJaccard()]

    for dataset_label in ALL_CV_DATASETS:
        print("Processing dataset " + str(dataset_label))
        hofs = flatten_hofs_for_dataset_cv(dataset_lab=dataset_label, main_labs=MAIN_LABS)
        solutions = []
        hof_names = []
        classes = []
        for alg_hofs in hofs:
            h_path = alg_hofs.path()
            if os.path.isdir(h_path):
                solutions.append(solutions_from_files(hof_dir=h_path))
                hof_names.append(alg_hofs.name())
                classes.append(alg_hofs.main_algorithm_label())
        for measure in fold_measures:
            print("Plotting measure " + measure.name())
            measure_nick = measure.nick()
            plot_path = SUMMARY_STAT_DIR + "/cv/" + dataset_label + "/" + measure_nick
            measure_vals = [measure.compute_measures(solutions=s) for s in solutions]
            barplot_with_std_to_file(path=plot_path, measures=measure_vals, bar_names=hof_names,
                                     label_y=measure.name(), classes=classes)
        for measure in global_measures:
            print("Plotting measure " + measure.name())
            measure_nick = measure.nick()
            plot_path = SUMMARY_STAT_DIR + "/cv/" + dataset_label + "/" + measure_nick
            measure_vals = [measure.compute_measure(solutions=s) for s in solutions]
            barplot_to_file(path=plot_path, bar_lengths=measure_vals, bar_names=hof_names,
                            label_y=measure.name(), classes=classes)
