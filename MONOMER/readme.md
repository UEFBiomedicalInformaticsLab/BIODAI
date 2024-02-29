# MONOMER

Publicly available source code and data for the project Multi-objective OptimizatioN Of bioMarkERs (MONOMER).
The project is described in Cattelani et al. [1].

The main program is in Python, we include also R scripts that were used to prepare the TCGA datasets.

The R script load_tcga_brca.R creates the mrna.csv and outcome.csv related to the breast TCGA dataset.
Similarly, the scripts load_tcga_kir.R, load_tcga_ov.R and luad_lusc.R create the csv files for the kidney,
ovary, and lung TCGA datasets.
In order to work the R scripts require an internet connection.

All the data files needed to run the tests are included in this repository, except for the mrna data because
it is too large for GitHub. All the results, including plots, are included in this repository,
but can also be generated again by a user launching the Python programs.

This project is licensed under the terms of the MIT license.

## Launching the programs

The main Python programs get in input an INI file with the configuration of a test. All the INI file used to produce
the paper results are in the directory work/setups.
A configuration file starts with [MVMOO_SETUP].

The most important parameters are the following.
- **dataset**. The name of the dataset to be used for internal-external cross-validation or as training for
external validation. Valid options include "brca" (TCGA breast), "luad_lusc" (TCGA lung),
"tcga_kir" (TCGA kidney with 4 classes), "tcga_kir3" (TCGA kidney with 3 classes), tcga_ov (TCGA ovary),
"swedish" (SCAN-B), "cptac3_sub_uq4" (CPTAC-3), "kid_gse152938d" (GSE152938), and "ext_ov2" (GSE102073).
- **mvmo_algorithm**. The name of the main algorithm. It can be "lasso_mo" (LASSO multi-objective), "guided_forward",
"rfe" (recursive feature elimination), and "classic_ga" (NSGA2 and the new variants).
- **objectives**. A string that specifies the objectives and if required also the inner model. For balanced accuracy and
leanness with naive Bayes as inner model it is "[["bal_acc", "naive_bayes"], "leanness"]". The inner model can be
naive_bayes", "RF" (random forest), "logistic", or "svm" (support vector machine).
The inner model is ignored by lasso_mo.
- **use_big_defaults**. Boolean parameter, the default is "false" and some parameters are set for a short test run.
When true these parameters are set for a long serious run. This must be set to true to reproduce our results.
- **external_dataset**. The name of the external dataset. The accepted values are the same of the parameter "dataset".
This parameter is ignored when running internal-external cross-validation.
- **cross_validation**. If true and if not running an external validation, the cross-validation is performed, and the
results saved.
- **final_optimization**. If true and if not running an external validation, the optimization on the whole dataset is
performed, and the results saved.
- **pop**. The size of the population for GA based algorithms.
- **generations**. A list of integers to support future extensions, we use lists of only one integer in these tests.
The value in the list is the number of generations used by the GA-based algorithms.
- **initial_features_strategy**. Strategy to extract the number of features when initializing a solution in GAs.
We always use "uniform" in our tests, so that the number of features is extracted with a uniform distribution.
Other two parameters are used to set the minimum and maximum number of features in an initial solution.
- **initial_features_min and initial_features_max**. Two numbers to specify the minimum and maximum number of features in an
initial solution.
- **sorting_strategy**. The sorting strategy to use before selection and tournament. It can be "crowding_distance_full"
for NSGA2 implied sorting or "crowding_distance_clone_index" to use the clone index as primary sorting criteria [2].
- **use_clone_repurposing**. A Boolean. If true clone repurposing is used [2].
- **logistic_max_iter**. The maximum number of iterations used by the logistic regression inner model. Ignored when using
another inner model.
- **feature_importance_categorical**. The strategy used to compute the feature importance for the GA-based algorithms.
With the default of "none" a uniform feature importance is used. With "lasso" the LASSO feature importance is used [2].
- **bitlist_mutation_operator**. With the default of "flip" a bit-flip operator is used. With "symm" the symmetric mutation
is used instead [2].
- **inner_n_folds**. The number of folds used inside the optimizer for evaluating the solutions. Ignored if the optimizer
does not use internal cross-validation. Defaults to 3.

The Python script py/run_with_setups.py is used to run the internal-external validation and the final optimization
(optimization on the whole dataset). It gets in input an INI setup file. An example of run from command line
(from inside the working directory "work") is
python ../py/run_with_setups.py setups/brca/test_small.ini

The Python script py/external_validator.py runs an external validation. It gets in input an INI setup file
and works in an analogous way as run_with_setups.py.

In py/plots there are scripts for creating summary plots and textual reports.
It is sufficient to run them without any parameter, and they
will search the work directory for the necessary test results and create the plots/reports.
The working directory must be the directory "work". The scripts are
- best_genes_plotter.py
- best_solutions_printer.py
- subplots_runner.py
- summary_statistics_plotter.py
Other similar scripts can be found in py/cattelani2023. They are
- best_biomarkers_table_plotter.py
- cattelani2023_gene_boxplots.py
- cattelani2023_pca_subplots.py
- hv_variety_subplotter.py
- performance_by_class_master_subplots.py
- performance_by_class_subplots.py
- subplots_for_inner_model.py
- subplots_runner_alltogether.py
- summary_statistics_subplotter.py

## Bibliography

[1] Luca Cattelani, Arindam Ghosh, Teemu Rintala, Vittorio Fortino. Improving biomarker selection for cancer subtype classification through multi-objective optimization. TechRxiv (2023), https://doi.org/10.36227/techrxiv.24321154.v2

[2] Luca Cattelani, Vittorio Fortino, Improved NSGA-II algorithms for multi-objective biomarker discovery, Bioinformatics, Volume 38, Issue Supplement_2, September 2022, Pages ii20–ii26, https://doi.org/10.1093/bioinformatics/btac463
