from individual.mv_feature_set_by_names import MVFeatureSetByNames
from load_omics_views import MRNA_NAME

PAM50_GENES = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20", "CDC6", "CDH3",
    "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160",
    "GRB7", "KIF2C", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK", "MIA", "MKI67", "MLPH",
    "MMP11", "MYBL2", "MYC", "NAT1", "NDC80", "NUF2", "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1",
    "SLC39A6", "TMEM45B", "TYMS", "UBE2C", "UBE2T", "ORC6L"]
PAM50_NAME = "PAM50"
PAM50 = MVFeatureSetByNames(features_by_view={MRNA_NAME: set(PAM50_GENES)}, name=PAM50_NAME)

GOLESTAN2024 = MVFeatureSetByNames(
    features_by_view={MRNA_NAME: {"CACNG4", "PKMYT1", "EPYC", "CHRNA6"}}, name="Golestan2024")
# From Golestan, A., Tahmasebi, A., Maghsoodi, N. et al.
# Unveiling promising breast cancer biomarkers: an integrative approach combining bioinformatics analysis
# and experimental verification. BMC Cancer 24, 155 (2024). https://doi.org/10.1186/s12885-024-11913-7


ONCOTYPE = MVFeatureSetByNames(
    features_by_view={MRNA_NAME: {
        "MKI67", "AURKA", "BIRC5", "CCNB1", "MYBL2", "GRB7", "ERBB2", "ESR1", "PGR", "BCL2", "SCUBE2",
        "MMP11", "CTSL2", "GSTM1", "CD68", "BAG1", "ACTB", "GAPDH", "GUSB", "RPLP0", "TFRC"
    }}, name="Oncotype DX")
# Genes from
# Melissa Krystel-Whittemore, Puay Hoon Tan, Hannah Y. Wen,
# Predictive and prognostic biomarkers in breast tumours,
# Pathology, Volume 56, Issue 2, 2024, Pages 186-191, ISSN 0031-3025,
# https://doi.org/10.1016/j.pathol.2023.10.014.
# (https://www.sciencedirect.com/science/article/pii/S0031302523003112)
# We assume GUS from the paper to be in fact GUSB.


LI2021 = MVFeatureSetByNames(
    features_by_view={MRNA_NAME: {
        "BIRC5", "XIAP", "HIF1A", "EPAS1", "NFE2L2", "MDM2", "MDM4", "TP53", "KRAS", "AKT1"
    }}, name="Li2021")
# Li, F., Aljahdali, I.A.M., Zhang, R. et al.
# Kidney cancer biomarkers and targets for therapeutics: survivin (BIRC5), XIAP, MCL-1, HIF1α, HIF2α, NRF2, MDM2, MDM4,
# p53, KRAS and AKT in renal cell carcinoma.
# J Exp Clin Cancer Res 40, 254 (2021). https://doi.org/10.1186/s13046-021-02026-1
# We are assuming HIF2α -> EPAS1, NRF2 -> NFE2L2, AKT -> AKT1


MEHTA2020 = MVFeatureSetByNames(
    features_by_view={MRNA_NAME: {
        "EGFR", "ALK", "ROS1", "BRAF", "KRAS", "AKT1", "ERBB2", "MAP2K1", "MET", "NRAS", "PIK3CA", "RET"
    }}, name="Mehta2020")
# Mehta, A., Vasudevan, S., Sharma, S.K. et al.
# Biomarker testing for advanced lung cancer by next-generation sequencing; a valid method to achieve a comprehensive
# glimpse at mutational landscape. Appl Cancer Res 40, 4 (2020). https://doi.org/10.1186/s41241-020-00089-8


MUINAO2019 = MVFeatureSetByNames(
    features_by_view={MRNA_NAME: {
        "MUC16", "FUT3", "EGFR", "CSF3", "CCL11", "IL2RA", "VCAM1", "MITF"
    }}, name="Muinao2019")
# Muinao, Thingreila, Hari Prasanna Deka Boruah, and Mintu Pal.
# "Multi-biomarker panel signature as the key to diagnosis of ovarian cancer." Heliyon 5.12 (2019).
# Assuming CA-125 -> MUC16, CA 19–9 -> FUT3, G-CSF -> CSF3, Eotaxin -> CCL11, IL-2R -> IL2RA, cVCAM -> VCAM1, MI -> MITF
