from cattelani2023.cattelani2023_utils import cattelani2023_internal_hofs, cattelani2023_external_hofs, CATTELANI2023_DIR
from plots.hofs_plotter.hofs_measure_plotter import HofsFoldMeasurePlotter
from plots.subplots_by_strategy import subplots_by_strategy
from saved_solutions.run_cross_hypervolume import RunCrossHypervolume
from saved_solutions.run_fold_jaccard import RunFoldJaccard
from saved_solutions.run_soft_cross_hypervolume import RunSoftCrossHypervolume

if __name__ == '__main__':
    ncols = 3
    row_measures = list((RunCrossHypervolume(), RunCrossHypervolume(), RunFoldJaccard()))
    plotters = [HofsFoldMeasurePlotter(measure=measure) for measure in row_measures]
    plot_path = CATTELANI2023_DIR + "/" + "hv_variety"
    i_hofs = cattelani2023_internal_hofs()
    e_hofs = cattelani2023_external_hofs()
    hofs = []
    for i in range(len(i_hofs)):
        hofs.append(i_hofs[i])
        hofs.append(e_hofs[i])
        hofs.append(i_hofs[i])
    subplots_by_strategy(
        hofs=hofs,
        plotter=plotters,
        save_path=plot_path,
        ncols=ncols,
        x_label=None, y_label=None,
        color_by_row=True)

    row_measures = list((RunSoftCrossHypervolume(), RunSoftCrossHypervolume(), RunFoldJaccard()))
    plotters = [HofsFoldMeasurePlotter(measure=measure) for measure in row_measures]
    plot_path = CATTELANI2023_DIR + "/" + "hv_variety_soft"
    subplots_by_strategy(
        hofs=hofs,
        plotter=plotters,
        save_path=plot_path,
        ncols=ncols,
        x_label=None, y_label=None,
        color_by_row=True)
