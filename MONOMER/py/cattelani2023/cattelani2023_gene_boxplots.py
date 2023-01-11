from input_data.brca_input_creator import BrcaInputCreator
from input_data.input_creators_archive import TCGA_KIR3_NICK, KID_GSE152938D_NICK, CPTAC3_SUB_UQ4_NICK, TCGA_OV_NICK, \
    EXT_OV2_NICK
from input_data.luad_lusc_input_creator import LuadLuscInputCreator
from input_data.swedish_input_creator import SwedishInputCreator
from cattelani2023.cattelani2023_utils import CATTELANI2023_DIR
from plots.plotter.gene_boxplotter import GeneBoxplotter
from plots.subplots import subplots
from setup.setup_utils import load_input_data


if __name__ == '__main__':
    input_data_blocks = [
        [load_input_data(dataset_name=BrcaInputCreator().nick()),
         load_input_data(dataset_name=SwedishInputCreator().nick())],
        [load_input_data(dataset_name=TCGA_KIR3_NICK),
         load_input_data(dataset_name=KID_GSE152938D_NICK)],
        [load_input_data(dataset_name=LuadLuscInputCreator().nick()),
         load_input_data(dataset_name=CPTAC3_SUB_UQ4_NICK)],
        [load_input_data(dataset_name=TCGA_OV_NICK),
         load_input_data(dataset_name=EXT_OV2_NICK)]
    ]
    genes = [
        ["BIRC5", "ESR1", "FOXC1", "MIA"],
        ["EGLN3", "TBC1D1"],
        ["ASB2", "C10orf55", "C11orf16", "CSTB", "ETS1", "FGL1", "GSTM4", "HLA-DQA2", "IFI30", "KIF2C", "KRT31", "MGLL",
         "MPZL2", "NLE1", "OSBP2", "PDXK", "PLA2G4F", "RPAP1", "SCAMP1", "SLC29A3", "UGDH", "UGT1A7"],
        ["CREB3L1", "GPR12", "HAVCR2", "NPNT", "PLEKHO1", "RFX8", "UBA7"]
    ]
    for block, gene_list in zip(input_data_blocks, genes):
        plotters = [GeneBoxplotter(input_data=block[0], feature_names=gene_list),
                    GeneBoxplotter(input_data=block[1], feature_names=gene_list)]
        subplots(plotters=plotters,
                 save_path=CATTELANI2023_DIR + "/" + "genes_" + block[0].nick() + "_" + block[1].nick(), ncols=1,
                 x_label="subtype",
                 y_label="expression",
                 x_stretch=3.0)
