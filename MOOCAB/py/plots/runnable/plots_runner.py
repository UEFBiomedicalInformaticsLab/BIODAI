from location_manager.path_utils import main_path_for_saves
from postprocessing.postprocessing import run_postprocessing_archive_cv_and_final

if __name__ == '__main__':

    MAIN_DIRECTORY = "tcga_thca3/mrna/leanness_svm_bal_acc/5_folds/"
    # MAIN_DIRECTORY = "luad_lusc/mrna/leanness_svm_bal_acc/5_folds/"
    OPTIMIZER_NICK = "NSGA2_k3_pop500_uni0-50_gen2000_CrowdCICR_c0.33_m1.0symm_(MV_lassoFI,none)"
    DIRECTORY = MAIN_DIRECTORY + OPTIMIZER_NICK + "/"

    main_hofs_dir = main_path_for_saves(base_path=MAIN_DIRECTORY, optimizer_nick=OPTIMIZER_NICK)

    JUHO_NSGA2_CH_NB = "juho_project/log_mrna/NB_bal_acc_root_leanness/5_folds/NSGA2_k3_pop500_uni0-50_gen1000_CrowdCI_c0.33_m1.0flip_(MV_lassoFI,none)"

    run_postprocessing_archive_cv_and_final(optimizer_dir="kidney_ihc_det_os/mrna/SKSurvCox_c-index_root_leanness/3_folds_x3/67445/adj_k3_NSGA3_k3_pop353_uni0-50_gen353_NSGA3CICR_c0.33_m1.0symm_adj_rSVR_NSGA3_k3_pop500_uni0-50_gen500_NSGA3CICR_c0.33_m1.0symm_(MV_lassoFI,MV_CoxFI)")
