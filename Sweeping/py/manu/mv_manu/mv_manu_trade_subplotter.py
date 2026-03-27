from manu.mv_manu.mv_manu_consts import MV_MANU_DIR, MV_MANU_N_COLS, MV_MANU_PLOT_SETUP
from manu.mv_manu.mv_manu_batteries import MV_MANU_HOFS
from plots.subplots_by_strategy import subtradeplots


if __name__ == '__main__':
    plot_path = MV_MANU_DIR + "/" + "trade_plots"
    subtradeplots(
        hofs=MV_MANU_HOFS,
        save_path=plot_path,
        ncols=MV_MANU_N_COLS,
        setup=MV_MANU_PLOT_SETUP)