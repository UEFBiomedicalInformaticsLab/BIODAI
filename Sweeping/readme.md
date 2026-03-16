# Sweeping*

Publicly available source code and data for the project Sweeping*.

The project is described in Cattelani and Fortino [5], soon to be published as a preprint.

The main program is in Python (version 2.9+), we include also R scripts that were used to prepare the TCGA datasets.

All the data files needed to run the tests are included in this repository. All the results, including plots, are
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
Valid options include...
- **mvmo_algorithm**. The name of the main algorithm. Select "classic_ga" for NSGA*, ...
- **objectives**. A string that specifies the objectives and if required also the inner model...
Survival analysis can be requested by inserting "["c-index", "Cox", "survival"]" in the list, e.g.
"["root_leanness", ["c-index", "Cox", "survival"]]".
- **use_big_defaults**. Boolean parameter, the default is "false" and some parameters are set for a short test run.
When true these parameters are set for a long serious run. This must be set to true to reproduce our results.
- **cross_validation**. If true and if not running an external validation, the k-fold cross-validation is performed, and the
results saved.
- **final_optimization**. If true and if not running an external validation, the optimization on the whole dataset is
performed, and the results saved.
- **pop**. The size of the population for GA based algorithms.
- **generations**. A list of integers to support future extensions, we use lists of only one integer in these tests.
The value in the list is the number of generations used by the GA-based algorithms.
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
- **logistic_max_iter**. The maximum number of iterations used by the logistic regression inner model. Ignored when using
another inner model.

The following is an example of setup file.
```
[MVMOO_SETUP]
dataset = kidney_ihc_det_os
mvmo_algorithm = adjusted
objectives = [["bal_acc", "naive_bayes"], "root_leanness", ["c-index", "Cox", "survival"]]
pop = 500
generations = [500]
use_big_defaults = true
fold_parallelism = false
initial_features_strategy = uniform
initial_features_min = 0
initial_features_max = 50
cross_validation = true
final_optimization = true
feature_importance_categorical = lasso
feature_importance_survival = cox
sorting_strategy = nsga3_clone_index
use_clone_repurposing = true
bitlist_mutation_operator = symm
outer_n_folds = 3
logistic_max_iter = 100
cv_repeats = 3
seed = 67445
adjuster_regressor = SVR
```

The Python script ___ is used to run the k-fold cross-validation and the final optimization
(optimization on the whole dataset). It gets in input an INI setup file. An example of run from command line
(from inside the working directory "work") is
```
python ...
```

By launching
```
python ../py/plot_all_from_batteries.py
```
from inside the "work" directory it is possible to produce all the summary tables and plots that aggregate multiple
runs by datasets and objectives. It automatically searches the work directory for the necessary test results and creates
the plots/tables.

The suggested list of package requirements is in the file requirements.txt.

The R script ...
In order to work the R scripts require an internet connection. These data files are already present in the work
directory, still the scripts are included for reproducibility.

## Program results

The results for a k-fold cross-validation are saved in a subdirectory of "work". The path is
composed by the name of the dataset, then the type of data ("mrna"), the objectives, the type of validation, the
random seed, and finally the type and parameters of the optimizer.

The directory of a k-fold cross-validation and/or final optimization contains the following items.
- **common_features_between_folds_top_k.png**
Average number of features in common between the folds when considering in each fold the top k more frequent features.
k increases from left to right. From top to bottom there is the passing of the generations. This plot is drawn only for
the GA based optimizers.
- **config.ini**
A copy of the configuration file that was used to set up the program.
- **feature_counts_*.csv**
A table for each fold, it reports every 100 generations the number of occurrences of each feature in the population.
Only for GA based optimizers.
- **folds.json**
The subdivision of the samples into folds.
- **log.png**
The max, min, and average fitness for each objective along the generations. Averages across the folds.
Only for GA based optimizers.
- **log.txt**
Textual log for the k-fold cross validation.
- **log_features.png**
For each generation, the number of features included in its population, and the number of features explored so far by
all present and past individuals. The values are averaged across the folds.
- **log_fold_*_features.png**
A plot for each fold. For each generation, the number of features included in its population, and the number of features
explored so far by all present and past individuals.
- **log_final.txt**
Textual log for the final optimization (optimization on the whole dataset).
- **log_fold_*.csv**
A csv table for each fold with the evolution of the fitnesses along the generations. For each objective, The max, min,
and average fitness. Only for GA based optimizers.
- **log_fold_*.png**
A file for each fold, listing the max, min, and average fitness for each objective along the generations.
Only for GA based optimizers.
- **log_fold_*.txt**
A textual log for each of the folds.
- **stability_between_folds.png**
Average pairwise stability between the folds of the selected features, across the generations.
Stability is measured by "weight overlap": the weight is the frequency of the gene in the population, scaled so that
the total sum of the weights is equal to 1. The overlap between two folds is computed by summing the elementwise min
weights [3]. Only for GA based optimizers.
- **stability_in_time.png**
Stability between populations 100 generations apart. Averaged across the folds. Only for GA based optimizers.
- **stability_in_time_fold_*.png**
Stability between populations 100 generations apart measured on each fold separately. Only for GA based optimizers.
- **stability_of_weights_between_folds_top_k.png**
Stability of features, measured by weight overlap [3], between the folds when considering in each fold the top k more
frequent features. k increases from left to right. From top to bottom there is the passing of the generations. This plot
is drawn only for the GA based optimizers.
- **stability_of_unions_between_folds_top_k.png**
Stability of features, measured by Dice score, between the folds when considering in each fold the top k
more frequent features. k increases from left to right. From top to bottom there is the passing of the generations.
This plot is drawn only for the GA based optimizers.
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
- **hofs/*/balanced_accuracy_by_class.png**
For each classification class the feature set size and balanced accuracy of the solutions. Plotted only if there is a
classification objective.
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
- **Best trade-off plots**
There is a best trade-off plot [3] for each hall of fame and pair of objectives.
- **hofs/*/precision_by_class.png**
For each classification class the feature set size and precision of the solutions. Plotted only if there is a
classification objective.
- **hofs/*/recall_by_class.png**
For each classification class the feature set size and recall of the solutions. Plotted only if there is a
classification objective.
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
- **hofs/*/confusion_matrix/**
If the setup includes a classification objective, this directory is filled with a csv file for each fold, representing
the confusion matrix of each solution. Order of the solutions in these files is consistent.

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

[5] Luca Cattelani, Vittorio Fortino. Genetic algorithms for multi-omic feature selection:
a comparative study in cancer survival analysis. To be published.

[6] Marcel Ramos, Ludwig Geistlinger, Sehyun Oh, Lucas Schiffer, Rimsha Azhar, Hanish Kodali, Ino
de Bruijn, Jianjiong Gao, Vincent J Carey, Martin Morgan, et al. Multiomic integration of public
oncology databases in bioconductor. JCO Clinical Cancer Informatics, 1:958–971, 2020.
