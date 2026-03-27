from plots.hofs_plotter.plot_setup import NO_LOG_PLOT_SETUP

MV_MANU_DIR = "mv_manu"
MV_MANU_N_COLS = 3
MV_MANU_PLOT_SETUP = NO_LOG_PLOT_SETUP.set_labels_map(
    labels_map=NO_LOG_PLOT_SETUP.labels_map().add("_mv","").add("kirc","KIRC").add("lgg","LGG").add("sarc","SARC").add(" NSGA3-CHS", ""))
MV_MANU_PLOT_SETUP = MV_MANU_PLOT_SETUP.set_decimals(decimals=None)
