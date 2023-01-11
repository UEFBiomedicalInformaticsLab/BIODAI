from input_data.brca_input_creator import BrcaInputCreator
from input_data.input_creators_archive import TCGA_KIR3_NICK, KID_GSE152938D_NICK, CPTAC3_SUB_UQ4_NICK, TCGA_OV_NICK, \
    EXT_OV2_NICK
from input_data.luad_lusc_input_creator import LuadLuscInputCreator
from input_data.swedish_input_creator import SwedishInputCreator
from cattelani2023.cattelani2023_utils import CATTELANI2023_DIR
from plots.pca_subplots import pca_subplots
from setup.setup_utils import load_input_data


DATASET_NAMES = [
    BrcaInputCreator().nick(),
    SwedishInputCreator().nick(),
    TCGA_KIR3_NICK,
    KID_GSE152938D_NICK,
    LuadLuscInputCreator().nick(),
    CPTAC3_SUB_UQ4_NICK,
    TCGA_OV_NICK,
    EXT_OV2_NICK
]


if __name__ == '__main__':
    pca_subplots(input_data=[load_input_data(dataset_name=name) for name in DATASET_NAMES],
                 save_path=CATTELANI2023_DIR + "/" + "pca")
