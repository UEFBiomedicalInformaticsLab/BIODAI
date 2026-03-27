from manu.mv_manu.mv_manu_consts import MV_MANU_N_COLS, MV_MANU_DIR, MV_MANU_PLOT_SETUP
from manu.mv_manu.mv_manu_batteries import MV_MANU_HOFS
from plots.hofs_plotter.hofs_measure_plotter import HofsFoldMeasurePlotter
from plots.subplots_by_strategy import subplots_by_strategy
from saved_solutions.run_measure.run_cross_hypervolume import RunCrossHypervolume
from saved_solutions.run_measure.run_pareto_delta import RunFoldParetoDelta

if __name__ == '__main__':
    ncols = MV_MANU_N_COLS
    fold_measures = list((RunCrossHypervolume(),RunFoldParetoDelta()))
    plot_setup = MV_MANU_PLOT_SETUP

    print("Plots for CV runs")
    hofs = MV_MANU_HOFS
    plotters = []
    for measure in fold_measures:
        plotters.append(HofsFoldMeasurePlotter(measure=measure, setup=plot_setup))
    for plotter in plotters:
        measure_name = plotter.measure_name()
        measure_nick = plotter.measure_nick()
        plot_path = MV_MANU_DIR + "/" + measure_nick
        print("Processing CV measure " + measure_name)
        subplots_by_strategy(
            hofs=hofs,
            plotter=plotter,
            save_path=plot_path,
            ncols=ncols,
            x_label=plot_setup.labels_map().apply(measure_name), y_label=None,
            labels_transformer=plot_setup.labels_map())
