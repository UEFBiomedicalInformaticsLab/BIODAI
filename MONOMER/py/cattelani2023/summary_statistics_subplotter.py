from cattelani2023.cattelani2023_utils import cattelani2023_external_hofs, cattelani2023_internal_hofs
from plots.hofs_plotter.hofs_measure_plotter import HofsFoldMeasurePlotter, HofsRunMeasurePlotter
from plots.subplots_by_strategy import subplots_by_strategy
from saved_solutions.run_best_dice import RunBestDice
from saved_solutions.run_cross_hypervolume import RunCrossHypervolume
from saved_solutions.run_fold_jaccard import RunFoldJaccard
from saved_solutions.run_soft_cross_hypervolume import RunSoftCrossHypervolume
from saved_solutions.run_test_hypervolume import RunTestHypervolume
from saved_solutions.run_weight_overlap import RunWeightOverlap

SUMMARY_STAT_DIR = "summary_stats"

if __name__ == '__main__':
    ncols = 2
    global_measures = list((RunWeightOverlap(), RunBestDice()))
    fold_measures = list((RunTestHypervolume(), RunCrossHypervolume(), RunSoftCrossHypervolume(), RunFoldJaccard()))

    print("Plots for external runs")
    hofs = cattelani2023_external_hofs()
    for measure in fold_measures:
        measure_name = measure.name()
        plot_path = SUMMARY_STAT_DIR + "/Cattelani2023/external/" + measure.nick()
        plotter = HofsFoldMeasurePlotter(measure=measure)
        print("Processing external measure " + measure_name)
        subplots_by_strategy(
            hofs=hofs,
            plotter=plotter,
            save_path=plot_path,
            ncols=ncols,
            x_label=None, y_label=measure_name)

    print("Plots for CV runs")
    hofs = cattelani2023_internal_hofs()
    plotters = []
    for measure in fold_measures:
        plotters.append(HofsFoldMeasurePlotter(measure=measure))
    for measure in global_measures:
        plotters.append(HofsRunMeasurePlotter(measure=measure))
    for plotter in plotters:
        measure_name = plotter.measure_name()
        measure_nick = plotter.measure_nick()
        plot_path = SUMMARY_STAT_DIR + "/Cattelani2023/CV/" + measure_nick
        print("Processing CV measure " + measure_name)
        subplots_by_strategy(
            hofs=hofs,
            plotter=plotter,
            save_path=plot_path,
            ncols=ncols,
            x_label=None, y_label=measure_name)
