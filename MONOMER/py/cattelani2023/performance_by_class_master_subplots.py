from input_data.input_creators_archive import KID_GSE152938D_NICK, CPTAC3_SUB_UQ4_NICK, EXT_OV2_NICK
from input_data.swedish_input_creator import SWEDISH_NICK
from cattelani2023.cattelani2023_utils import CATTELANI2023_DIR
from plots.plot_labels import ALL_INNER_LABS, NSGA2_LAB, NSGA2_CHS_LAB, TCGA_BRCA_LAB, SVM_LAB, TCGA_KI3_LAB, RF_LAB, \
    NB_LAB, TCGA_LU_LAB, TCGA_OV_LAB, NSGA2_CH_LAB
from plots.plotter.performance_by_class_plotter import PerformanceByClassPlotter
from plots.saved_external_val import SavedExternalVal
from plots.subplots_by_strategy import subplots_by_strategy


if __name__ == '__main__':
    inner_labs = ALL_INNER_LABS
    main_labs = [NSGA2_LAB, NSGA2_CH_LAB, NSGA2_CHS_LAB]

    ncols = len(main_labs)
    n_inner = len(inner_labs)
    external_vals = [
        SavedExternalVal(
            internal_label=TCGA_BRCA_LAB, external_nick=SWEDISH_NICK, main_labs=main_labs, inner_labs=[SVM_LAB]),
        SavedExternalVal(
            internal_label=TCGA_KI3_LAB, external_nick=KID_GSE152938D_NICK, main_labs=main_labs, inner_labs=[RF_LAB]),
        SavedExternalVal(
            internal_label=TCGA_LU_LAB, external_nick=CPTAC3_SUB_UQ4_NICK, main_labs=main_labs, inner_labs=[NB_LAB]),
        SavedExternalVal(
            internal_label=TCGA_OV_LAB, external_nick=EXT_OV2_NICK, main_labs=main_labs, inner_labs=[SVM_LAB])
    ]
    plot_path = CATTELANI2023_DIR + "/" + "performance_by_class_master"
    external_hofs = []
    for ext in external_vals:
        for e in ext.nested_hofs():
            for f in e:
                external_hofs.append([f])
    plotter = [PerformanceByClassPlotter()] * len(external_hofs)
    plotter[1] = PerformanceByClassPlotter(vertical_lines=(4.0,))
    plotter[3] = PerformanceByClassPlotter(vertical_lines=(2.0,))
    plotter[8] = PerformanceByClassPlotter(vertical_lines=(22.0,))
    plotter[9] = PerformanceByClassPlotter(vertical_lines=(7.0,))
    subplots_by_strategy(
        hofs=external_hofs,
        plotter=plotter,
        save_path=plot_path,
        ncols=ncols,
        x_label="number of features", y_label="balanced accuracy")
