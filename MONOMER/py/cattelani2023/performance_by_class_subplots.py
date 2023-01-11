from cattelani2023.cattelani2023_utils import CATTELANI2023_DIR
from plots.archives.shallow_saved_hofs_archive_external import cattelani2023_external_validations
from plots.plot_labels import ALL_INNER_LABS, NSGA2_LAB, NSGA2_CHS_LAB, NSGA2_CH_LAB
from plots.plotter.performance_by_class_plotter import PerformanceByClassPlotter
from plots.subplots_by_strategy import subplots_by_strategy


if __name__ == '__main__':
    inner_labs = ALL_INNER_LABS
    main_labs = [NSGA2_LAB, NSGA2_CH_LAB, NSGA2_CHS_LAB]

    ncols = 3
    n_inner = len(inner_labs)
    plotter = PerformanceByClassPlotter()
    for i in range(n_inner):
        inner = inner_labs[i]
        print("Processing inner model " + inner)
        plot_path = CATTELANI2023_DIR + "/" + "performance_by_class_" + inner
        print("Processing external validations")
        external_hofs = []
        for ext in cattelani2023_external_validations(main_labs=main_labs, inner_labs=[inner]):
            for e in ext.nested_hofs():
                for f in e:
                    external_hofs.append([f])
        subplots_by_strategy(
            hofs=external_hofs,
            plotter=plotter,
            save_path=plot_path,
            ncols=ncols,
            x_label="number of features", y_label="balanced accuracy")
