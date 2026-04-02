# Sweeping*

Publicly available source code and data for the project Sweeping*.

The project is described in Cattelani and Fortino [5].

The main program is in Python (version 2.9+), we include also R scripts that were used to prepare the TCGA datasets.

All the data files needed to run the tests are included in this repository.
To save space, "work/kirc_mv/input/log_mrna.csv" is zipped in "work/kirc_mv/input/log_mrna.zip"
and must be unzipped before running the program on the TCGA-kirc dataset. Analogously, the same
applies for "work/lgg_mv/input/log_mrna.csv" and "work/sarc_mv/input/log_mrna.csv".
All the results, including plots, are
included in this repository, but can also be generated again by a user launching the Python programs.
Information on the datasets is in Cattelani and Fortino [5].

Experiments were addressed by using k-fold CV on TCGA [4] data.
TCGA transcriptomic datasets were downloaded with the curatedTCGAData R-package version 2.0.1
from assays of type RNASeq2GeneNorm [6].

This project is licensed under the terms of the MIT license.

## How to start the programs

The main Python programs receive in input an INI file with the configuration of a test.
All the INI files used to produce the paper results are in the directory "work/setups".
A configuration file starts with "[MVMOO_SETUP]".

The most important parameters, including all the parameters needed to replicate our results, are the following.
- **dataset.** The name of the dataset to be used for k-fold cross-validation.
Valid options include "kirc_mv", "lgg_mv", and "sarc_mv".
- **mvmo_algorithm**. The name of the main algorithm. Select "classic_ga" for NSGA* (NSGA3-CHS is
a declination of NSGA* [1], it uses the concatenated method when presented with multi-view data),
"sweeping_ga" for resampled sweeping or resampled sweeping with tuning,
"csweeping_ga" for concatenated sweeping, or "lcsweeping_ga" for lean concatenated sweeping.
- **objectives**. A string that specifies the objectives and if required also the inner model.
"root_leanness" adds the root-leanness to the objectives.
Survival analysis can be requested by inserting "["c-index", "Cox", "survival"]" in the list.
The tests described in Cattelani and Fortino [5] use the following string to define the objectives:
"[["c-index", "Cox", "survival"], "root_leanness"]".
- **use_big_defaults**. Boolean parameter, the default is "false" and some parameters are set for a short test run.
When true these parameters are set for a long serious run. This must be set to true to reproduce our results.
- **cross_validation**. If true and if not running an external validation, the k-fold cross-validation is performed,
and the results saved.
- **final_optimization**. If true and if not running an external validation, the optimization on the whole dataset is
performed, and the results saved.
- **pop**. The size of the population for GA based algorithms.
- **sweeping_generations**. A list of integers, e.g. "[50,50,50]".
The values in the list are the number of generations used for each sweep of the Sweeping* algorithm.
Defaults to the empty list.
- **concatenated_generations**. The number of concatenated generations to be executed after the sweeps in the
tuning phase. Set this to 0 to remove the tuning phase.
- **initial_features_strategy**. Strategy to extract the number of features when initializing a solution in GAs.
We always use "uniform" in our tests, so that the number of features is extracted with a uniform distribution.
Other two parameters are used to set the minimum and maximum number of features in an initial solution.
- **initial_features_min** and **initial_features_max**. Two numbers to specify the minimum and
maximum number of features in an initial solution.
- **sorting_strategy**. The sorting strategy to use before selection and tournament. It can be "crowding_distance_full"
for NSGA2 implied sorting, "crowding_distance_clone_index" to use the clone index as primary sorting criteria [3],
"nsga3" for NSGA3 implied sorting, or
"nsga3_clone_index" to use NSGA3 implied sorting with clone index as primary sorting criteria.
- **use_clone_repurposing**. A Boolean. Defaults to false. If true clone repurposing [3] is used.
- **bitlist_mutation_operator**. With the default of "flip" a bit-flip operator is used.
With "symm" the symmetric mutation is used instead.
- **feature_importance_categorical**. The strategy used to take classification into account for computing the feature
importance [3] for the GA-based algorithms. With the default of "none" a uniform feature importance is used.
With "lasso" the LASSO feature importance is used.
- **feature_importance_survival**. The strategy used to take survival into account for computing the feature
importance [3] for the GA-based algorithms. With the default of "none" a uniform feature importance is used.
With "cox" the coefficients of an adaptively l1 regularized Cox are used to compute the feature importance.
- **inner_n_folds**. The number of folds used inside the optimizer for evaluating the solutions. Ignored if the optimizer
does not use internal cross-validation. Defaults to 3.
- **outer_n_folds**. The number of folds used when performing k-fold cross-validation. Defaults to 5.
- **cv_repeats**. The number of repetitions of the k-fold cross-validation. Defaults to 1.
- **fold_parallelism**. A Boolean. Defaults to true. If true the folds of a k-fold cross-validation are run in parallel.
It is suggested to disable this parallelism when running survival analysis because its parallel execution is not
supported on every system configuration.
- **seed**. Integer value used to initialize the pseudo-random number generation. Defaults to 48723.

The following is an example of setup file.
```
[MVMOO_SETUP]
dataset = lgg_mv
mvmo_algorithm = sweeping_ga
objectives = [["c-index", "Cox", "survival"], "root_leanness"]
views_to_use = ["log_mirna", "log_mrna", "clinic"]
feature_importance_categorical = lasso
feature_importance_survival = cox
cross_validation = true
final_optimization = true
use_big_defaults = true
fold_parallelism = false
pop = 500
sweeping_generations = [50,50]
concatenated_generations = 100
initial_features_strategy = uniform
initial_features_min = 0
initial_features_max = 50
sorting_strategy = nsga3_clone_index
use_clone_repurposing = true
bitlist_mutation_operator = symm
outer_n_folds = 5
inner_n_folds = 3
```

The Python script py/biodai_cv.py is used to run the k-fold cross-validation and the final optimization
(optimization on the whole dataset). It gets in input an INI setup file. An example of run from command line
(from inside the working directory "work") is
```
python ../py/biodai_cv.py setups/kirc_mv/survival_test.ini
```

By launching
```
python ../py/plot_all_from_batteries.py
```
from inside the "work" directory it is possible to produce all the summary tables and plots that aggregate multiple
runs by datasets and objectives. It automatically searches the work directory for the necessary test results and creates
the plots/tables.

## Program results

The results for a k-fold cross-validation are saved in a subdirectory of "work". The path is
composed by the name of the dataset, then the type of data ("mrna"), the objectives, the type of validation, the
random seed, and finally the type and parameters of the optimizer.

The directory of a k-fold cross-validation and/or final optimization contains the following items.
- **config.ini**
A copy of the configuration file that was used to set up the program.
- **folds.json**
The subdivision of the samples into folds.
- **log.txt**
Textual log for the k-fold cross validation.
- **log_final.txt**
Textual log for the final optimization (optimization on the whole dataset).
- **log_fold_*.txt**
A textual log for each of the folds.
- **workers_log.txt**
The program uses a number of workers (by default the number of cores detected in the system) to evaluate individuals
in parallel. This is the log for the workers related to the final optimization. It is usually empty and serves mainly
for debugging.
- **workers_log_fold_*.txt**
The log for the workers of a given fold in k-fold cross-validation. It is usually empty and serves mainly for debugging.
- **objective_pairs/**
This directory contains plots of solution fitnesses by considering the objectives 2 at a time. There are plots for each
considered hall of fame (Pareto, last population, top 50/100 by sum of fitnesses). Plots for each fold separately and
with all folds together. Plots with names ending in "ci" show the 95% confidence intervals of the fitnesses,
where available.
- **hofs/**
This directory contains subdirectories with results for the considered halls of fame (Pareto, last population,
top 50/100 by sum of fitnesses). The result files for the halls of fame are described below.
- **hofs/*/common_features.png**
Average number of features in common between the folds considering in each of them the k most frequent features.
k increases from left to right.
- **hofs/*/dice.png**
Average Dice score between the folds considering in each of them the k most frequent features.
k increases from left to right.
- **hofs/*/hof_weight_stability.png**
Average weight overlap [3] between the folds considering in each of them the k most frequent features.
k increases from left to right.
- **hofs/*/jaccard.png**
Average Jaccard index between the folds considering in each of them the k most frequent features.
k increases from left to right.
- **hofs/*/solution_ci_max_final.csv**
Higher endpoints of confidence intervals for the fitnesses of the solutions obtained from the final optimization.
Order of the solutions in these files is consistent.
- **hofs/*/solution_ci_max_fold_*.csv**
Higher endpoints of confidence intervals for the fitnesses of the solutions obtained from the optimization in a fold
of the k-fold cross-validation. Order of the solutions in these files is consistent.
- **hofs/*/solution_ci_min_final.csv**
Lower endpoints of confidence intervals for the fitnesses of the solutions obtained from the final optimization.
Order of the solutions in these files is consistent.
- **hofs/*/solution_ci_min_fold_*.csv**
Lower endpoints of confidence intervals for the fitnesses of the solutions obtained from the optimization in a fold
of the k-fold cross-validation. Order of the solutions in these files is consistent.
- **hofs/*/solution_features_fold_*.csv**
The features selected by the solutions of a fold, a solution for each row. Order of the solutions in these files is
consistent.
- **hofs/*/solution_features_fold_final.csv**
The features selected by the solutions of the final optimization, a solution for each row. Order of the solutions in
these files is consistent.
- **hofs/*/solution_fitnesses_final.csv**
The fitnesses of the solutions of the final optimization, a solution for each row. Order of the solutions in
these files is consistent.
- **hofs/*/solution_fitnesses_fold_*.csv**
The fitnesses of the solutions of a fold, a solution for each row. Order of the solutions in these files is
consistent.
- **hofs/*/solution_std_devs_final.csv**
The fitness standard deviations of the solutions of the final optimization, a solution for each row. Order of the
solutions in these files is consistent.
- **hofs/*/solution_std_devs_fold_*.csv**
The fitness standard deviations of the solutions of a fold, a solution for each row. Order of the solutions in these
files is consistent.
- **hofs/*/validation_registry.json**
A JSON file with the numerical values of summary statistics like the cross hypervolume [2] and the Pareto delta [1].
Statistics are saved in this file in order to compute them only once.
- **hofs/*/view_counts_*.png**
For each objective there is a plot showing the average number of features of the solutions for each value of the
fitness. The fitness used is the one estimated by the optimizer.

## Summary results

Starting from the "work" directory, it is possible to find also analyses that take into account multiple
program runs at the same time. There is one battery of this kind of analyses for each combination of
dataset, objectives and validation type (k-fold CV or external).
- **work/summary_stats/***
For each battery, barplots that compare the algorithms according to different metrics. E.g. Cross hypervolume
or Pareto delta.
- **work/summary_stats/*/baseline_best_comparison.png**
A comparison of performance between the best solutions found with just clinic
and the best solutions found considering all the combinations of views and Sweeping* configuration (according to CHV).
All the folds are shown together.
- **work/summary_stats/cv/*/best_hof.txt**
This file is present when the battery is evaluated with k-fold CV.
It is a list of the biomarkers found by the best algorithm.
The best algorithm is chosen according to the CHV measured with k-fold CV.
The biomarkers are then computed running the best algorithm on the whole dataset.

## Bibliography

[1] Luca Cattelani, Vittorio Fortino. Dual-stage optimizer for systematic overestimation
adjustment applied to multi-objective genetic algorithms for biomarker selection.
Briefings in Bioinformatics, Volume 26, Issue 1, January 2025, bbae674.
URL https://doi.org/10.1093/bib/bbae674

[2] Luca Cattelani, Arindam Ghosh, Teemu Rintala, Vittorio Fortino.
A Comprehensive Evaluation Framework for Benchmarking Multi-Objective Feature Selection
in Omics-Based Biomarker Discovery.
In IEEE/ACM Transactions on Computational Biology and Bioinformatics,
vol. 21, no. 6, pp. 2432-2446, Nov.-Dec. 2024.
URL https://doi.org/10.1109/TCBB.2024.3480150

[3] Luca Cattelani, Vittorio Fortino. Improved NSGA-II algorithms for multi-objective biomarker discovery.
Bioinformatics, Volume 38, Issue Supplement_2, September 2022, Pages ii20–ii26.
URL https://doi.org/10.1093/bioinformatics/btac463

[4] Carolyn Hutter, Jean Claude Zenklusen. The cancer genome atlas: creating lasting value beyond
its data. Cell, 173(2):283–285, 2018.

[5] Luca Cattelani, Vittorio Fortino.
“Genetic algorithms for multi-omic feature selection: A comparative study in cancer survival analysis,”
arXiv preprint, arXiv:2604.00065, 2026. doi: 10.48550/arXiv.2604.00065.

[6] Marcel Ramos, Ludwig Geistlinger, Sehyun Oh, Lucas Schiffer, Rimsha Azhar, Hanish Kodali, Ino
de Bruijn, Jianjiong Gao, Vincent J Carey, Martin Morgan, et al. Multiomic integration of public
oncology databases in bioconductor. JCO Clinical Cancer Informatics, 1:958–971, 2020.
