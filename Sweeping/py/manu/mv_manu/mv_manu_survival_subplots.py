from input_data.input_creator.input_creators_archive import KIRC_MV_NICK, LGG_MV_NICK, SARC_MV_NICK
from manu.mv_manu.mv_manu_consts import MV_MANU_DIR, MV_MANU_N_COLS
from plots.survival_subplots import survival_subplots
from setup.setup_utils import load_input_data

DATASET_NAMES = [
    KIRC_MV_NICK,
    LGG_MV_NICK,
    SARC_MV_NICK
]


if __name__ == '__main__':
    survival_subplots(
        input_data=[load_input_data(dataset_name=name) for name in DATASET_NAMES],
        save_path=MV_MANU_DIR + "/" + "survival",
        n_cols=MV_MANU_N_COLS)