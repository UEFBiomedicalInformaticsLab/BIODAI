from plots.hofs_plotter.hofs_measure_plotter import HofsFoldMeasurePlotter
from plots.subplots_by_strategy import subplots_by_strategy
from saved_solutions.run_measure.run_cross_hypervolume import RunCrossHypervolume
from survival_plots.survival_plot_utils import survival_external_hofs

SUMMARY_STAT_DIR = "summary_stats"

if __name__ == '__main__':
    ncols = 2
    global_measures = list()
    fold_measures = list((RunCrossHypervolume(),))

    print("Plots for external runs")
    print("TODO: check if root leanness is handled correctly.")
    hofs = survival_external_hofs()
    for measure in fold_measures:
        measure_name = measure.name()
        plot_path = SUMMARY_STAT_DIR + "/survival/external/" + measure.nick()
        plotter = HofsFoldMeasurePlotter(measure=measure)
        print("Processing external measure " + measure_name)
        subplots_by_strategy(
            hofs=hofs,
            plotter=plotter,
            save_path=plot_path,
            ncols=ncols,
            x_label=None, y_label=measure_name)
