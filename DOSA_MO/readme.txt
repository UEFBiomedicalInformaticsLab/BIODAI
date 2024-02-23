#DOSA-MO

Publicly available source code and data for the project DOSA-MO (Dual-stage Optimizer for Systematic overestimation
Adjustment in Multi-Objective problems).

The project is described in Cattelani and Fortino [1].

The main program is in Python, we include also R scripts that were used to prepare the TCGA datasets.

All the data files needed to run the tests are included in this repository. All the results, including plots, are
included in this repository, but can also be generated again by a user launching the Python programs.
TCGA breast and SCAN-B mrna data files are zipped because they are too large for GitHub. Unzipping the files to the same
directory is needed in order to use them. TCGA breast data is in "work/brca/input/mrna.zip". SCAN-B data is in a
split zip archive in the files "work/swedish/input/mrna.zip.00*". Information on the datasets and our preprocessing
is in Cattelani and Fortino [1].

The main Python programs receive in input an INI file with the configuration of a test. All the INI file used to produce
the paper results are in the directory "work/setups".
A configuration file starts with "[MVMOO_SETUP]".

The most important parameters, including all the parameters needed to replicate our results, are the following.
- dataset. The name of the dataset to be used for k-fold cross-validation or as training for external validation.
Valid options include "brca" (TCGA breast), "swedish" (SCAN-B),
"kidney_ihc_det" (TCGA kidney with genes favorable for immunohistochemistry),
"kidney_ihc_det_os" (TCGA kidney with genes favorable for immunohistochemistry, including survival data),
and "custom". The preparation of the cancer datasets is described in Cattelani and Fortino [1].
Selecting the "custom" option a user can provide its own dataset.
- mvmo_algorithm. The name of the main algorithm. Select "classic_ga" for NSGA*, "adjusted" for DOSA-MO wrapping NSGA*.
- adjuster_regressor. The regression model used by DOSA-MO to predict the overestimation. Ignored by the other main
algorithms. The options include "zero" (predicts always 0), "dummy" (weighed average), "ptree" (pruned decision tree),
"RFReg" (random forest regression), "SVR" (support vector regression), and "rSVR" (SVR with optimized regularization).
- objectives. A string that specifies the objectives and if required also the inner model. For balanced accuracy and
root-leanness with naive Bayes as inner model it is "[["bal_acc", "naive_bayes"], "root_leanness"]".
Supported inner models for classification include "naive_bayes", "svm" (support vector machine), and "logistic".
Survival analysis can be requested by inserting "["c-index", "Cox", "survival"]" in the list, e.g.
"[["bal_acc", "svm"], "root_leanness", ["c-index", "Cox", "survival"]]".
- use_big_defaults. Boolean parameter, the default is "false" and some parameters are set for a short test run.
When true these parameters are set for a long serious run. This must be set to true to reproduce our results.
- external_dataset. The name of the external dataset. The accepted values are the same of the parameter "dataset".
This parameter is ignored when running k-fold cross-validation.
- cross_validation. If true and if not running an external validation, the k-fold cross-validation is performed, and the
results saved.
- final_optimization. If true and if not running an external validation, the optimization on the whole dataset is
performed, and the results saved.
- pop. The size of the population for GA based algorithms.
- generations. A list of integers to support future extensions, we use lists of only one integer in these tests.
The value in the list is the number of generations used by the GA-based algorithms.
- initial_features_strategy. Strategy to extract the number of features when initializing a solution in GAs.
We always use "uniform" in our tests, so that the number of features is extracted with a uniform distribution.
Other two parameters are used to set the minimum and maximum number of features in an initial solution.
- initial_features_min and initial_features_max. Two numbers to specify the minimum and maximum number of features in an
initial solution.
- sorting_strategy. The sorting strategy to use before selection and tournament. It can be "crowding_distance_full"
for NSGA2 implied sorting, "crowding_distance_clone_index" to use the clone index as primary sorting criteria [3], or
"nsga3_clone_index" to use NSGA3 implied sorting with clone index as primary sorting criteria.
- use_clone_repurposing. A Boolean. Defaults to false. If true clone repurposing [3] is used.
- feature_importance_categorical. The strategy used to take classification into account for computing the feature
importance [3] for the GA-based algorithms. With the default of "none" a uniform feature importance is used.
With "lasso" the LASSO feature importance is used.
- feature_importance_survival. The strategy used to take survival into account for computing the feature
importance [3] for the GA-based algorithms. With the default of "none" a uniform feature importance is used.
With "cox" the coefficients of an adaptively l1 regularized Cox are used to compute the feature importance.
- inner_n_folds. The number of folds used inside the optimizer for evaluating the solutions. Ignored if the optimizer
does not use internal cross-validation. Defaults to 3.
- outer_n_folds. The number of folds used when performing k-fold cross-validation. Defaults to 5.
- cv_repeats. The number of repetitions of the k-fold cross-validation. Defaults to 1.
- fold_parallelism. A Boolean. Defaults to true. If true the folds of a k-fold cross-validation are run in parallel.
It is suggested to disable this parallelism when running survival analysis because its parallel execution is not
supported on every system configuration.
- seed. Integer value used to initialize the pseudo-random number generation. Defaults to 48723.
- logistic_max_iter. The maximum number of iterations used by the logistic regression inner model. Ignored when using
another inner model.

The Python script py/run_with_setups.py is used to run the k-fold cross-validation and the final optimization
(optimization on the whole dataset). It gets in input an INI setup file. An example of run from command line
(from inside the working directory "work") is
python ../py/run_with_setups.py setups/kidney_ihc_det/bal_acc/adj_svr_nsga3_chs_short.ini

The Python script py/external_validator.py runs an external validation. It gets in input an INI setup file
and works in an analogous way as run_with_setups.py.

By launching
python ../py/plot_all_from_batteries.py
from inside the "work" directory it is possible to produce all the summary tables and plots that aggregate multiple
runs by datasets and objectives. It automatically searches the work directory for the necessary test results and creates
the plots/tables.

The results for a k-fold cross-validation or an external validation are saved in a subdirectory of "work". The path is
composed by the name of the dataset, then the type of data ("mrna"), the objectives, the type of validation, the
random seed, and finally the type and parameters of the optimizer.

The directory of a k-fold cross-validation and/or final optimization contains the following items.
- common_features_between_folds_top_k.png
Average number of features in common between the folds when considering in each fold the top k more frequent features.
k increases from left to right. From top to bottom there is the passing of the generations. This plot is drawn only for
the GA based optimizers.
- config.ini
A copy of the configuration file that was used to set up the program.
- feature_counts_*.csv
A table for each fold, it reports every 100 generations the number of occurrences of each feature in the population.
Only for GA based optimizers.
- folds.json
The subdivision of the samples into folds.
- log.png
The max, min, and average fitness for each objective along the generations. Averages across the folds.
Only for GA based optimizers.
- log.txt
Textual log for the k-fold cross validation.
- log_features.png
For each generation, the number of features included in its population, and the number of features explored so far by
all present and past individuals. The values are averaged across the folds.
- log_fold_*_features.png
A plot for each fold. For each generation, the number of features included in its population, and the number of features
explored so far by all present and past individuals.
- log_final.txt
Textual log for the final optimization (optimization on the whole dataset).
- log_fold_*.csv
A csv table for each fold with the evolution of the fitnesses along the generations. For each objective, The max, min,
and average fitness. Only for GA based optimizers.
- log_fold_*.png
A file for each fold, listing the max, min, and average fitness for each objective along the generations.
Only for GA based optimizers.
- log_fold_*.txt
A textual log for each of the folds.
- stability_between_folds.png
Average pairwise stability between the folds of the selected features, across the generations.
Stability is measured by "weight overlap": the weight is the frequency of the gene in the population, scaled so that
the total sum of the weights is equal to 1. The overlap between two folds is computed by summing the elementwise min
weights [3]. Only for GA based optimizers.
- stability_in_time.png
Stability between populations 100 generations apart. Averaged across the folds. Only for GA based optimizers.
- stability_in_time_fold_*.png
Stability between populations 100 generations apart measured on each fold separately. Only for GA based optimizers.
- stability_of_weights_between_folds_top_k.png
Stability of features, measured by weight overlap [3], between the folds when considering in each fold the top k more
frequent features. k increases from left to right. From top to bottom there is the passing of the generations. This plot
is drawn only for the GA based optimizers.
- stability_of_unions_between_folds_top_k.png
Stability of features, measured by Dice score, between the folds when considering in each fold the top k
more frequent features. k increases from left to right. From top to bottom there is the passing of the generations.
This plot is drawn only for the GA based optimizers.
- workers_log.txt
The program uses a number of workers (by default the number of cores detected in the system) to evaluate individuals
in parallel. This is the log for the workers related to the final optimization. It is usually empty and serves mainly
for debugging.
- workers_log_fold_*.txt
The log for the workers of a given fold in k-fold cross-validation. It is usually empty and serves mainly for debugging.
- objective_pairs/
This directory contains plots of solution fitnesses by considering the objectives 2 at a time. There are plots for each
considered hall of fame (Pareto, last population, top 50/100 by sum of fitnesses). Plots for each fold separately and
with all folds together. Plots with names ending in "ci" show the 95% confidence intervals of the fitnesses,
where available.
- hofs/
This directory contains subdirectories with results for the considered halls of fame (Pareto, last population,
top 50/100 by sum of fitnesses). The result files for the halls of fame are described below.
- hofs/*/balanced_accuracy_by_class.png
For each classification class the feature set size and balanced accuracy of the solutions. Plotted only if there is a
classification objective.
- hofs/*/common_features.png
Average number of features in common between the folds considering in each of them the k most frequent features.
k increases from left to right.
- hofs/*/dice.png
Average Dice score between the folds considering in each of them the k most frequent features.
k increases from left to right.
- hofs/*/hof_weight_stability.png
Average weight overlap [3] between the folds considering in each of them the k most frequent features.
k increases from left to right.
- hofs/*/jaccard.png
Average Jaccard index between the folds considering in each of them the k most frequent features.
k increases from left to right.
- Best trade-off plots
There is a best trade-off plot [3] for each hall of fame and pair of objectives.
- hofs/*/precision_by_class.png
For each classification class the feature set size and precision of the solutions. Plotted only if there is a
classification objective.
- hofs/*/recall_by_class.png
For each classification class the feature set size and recall of the solutions. Plotted only if there is a
classification objective.
- hofs/*/solution_ci_max_final.csv
Higher endpoints of confidence intervals for the fitnesses of the solutions obtained from the final optimization.
Order of the solutions in these files is consistent.
- hofs/*/solution_ci_max_fold_*.csv
Higher endpoints of confidence intervals for the fitnesses of the solutions obtained from the optimization in a fold
of the k-fold cross-validation. Order of the solutions in these files is consistent.
- hofs/*/solution_ci_min_final.csv
Lower endpoints of confidence intervals for the fitnesses of the solutions obtained from the final optimization.
Order of the solutions in these files is consistent.
- hofs/*/solution_ci_min_fold_*.csv
Lower endpoints of confidence intervals for the fitnesses of the solutions obtained from the optimization in a fold
of the k-fold cross-validation. Order of the solutions in these files is consistent.
- hofs/*/solution_features_fold_*.csv
The features selected by the solutions of a fold, a solution for each row. Order of the solutions in these files is
consistent.
- hofs/*/solution_features_fold_final.csv
The features selected by the solutions of the final optimization, a solution for each row. Order of the solutions in
these files is consistent.
- hofs/*/solution_fitnesses_final.csv
The fitnesses of the solutions of the final optimization, a solution for each row. Order of the solutions in
these files is consistent.
- hofs/*/solution_fitnesses_fold_*.csv
The fitnesses of the solutions of a fold, a solution for each row. Order of the solutions in these files is
consistent.
- hofs/*/solution_std_devs_final.csv
The fitness standard deviations of the solutions of the final optimization, a solution for each row. Order of the
solutions in these files is consistent.
- hofs/*/solution_std_devs_fold_*.csv
The fitness standard deviations of the solutions of a fold, a solution for each row. Order of the solutions in these
files is consistent.
- hofs/*/validation_registry.json
A JSON file with the numerical values of summary statistics like the cross hypervolume [2] and the Pareto delta [1].
Statistics are saved in this file in order to compute them only once.
- hofs/*/view_counts_*.png
For each objective there is a plot showing the average number of features of the solutions for each value of the
fitness. The fitness used is the one estimated by the optimizer.
- hofs/*/confusion_matrix/
If the setup includes a classification objective, this directory is filled with a csv file for each fold, representing
the confusion matrix of each solution. Order of the solutions in these files is consistent.

The directory of an external validation contains the following items.
- log_final.txt
Textual log.
- config.ini
A copy of the configuration file that was used to set up the program.
- workers_log.txt
The program uses a number of workers (by default the number of cores detected in the system) to evaluate individuals
in parallel. This is the log for the workers. It is usually empty and serves mainly for debugging.
- objective_pairs/
This directory contains plots of solution fitnesses by considering the objectives 2 at a time. There are plots for each
considered hall of fame (Pareto, last population, top 50/100 by sum of fitnesses).
- hofs/
This directory contains subdirectories with results for the considered halls of fame (Pareto, last population,
top 50/100 by sum of fitnesses). The result files for the halls of fame are described below.
- hofs/*/balanced_accuracy_by_class.png
For each classification class the feature set size and balanced accuracy of the solutions. Plotted only if there is a
classification objective.
- hofs/*/precision_by_class.png
For each classification class the feature set size and precision of the solutions. Plotted only if there is a
classification objective.
- hofs/*/recall_by_class.png
For each classification class the feature set size and recall of the solutions. Plotted only if there is a
classification objective.
- hofs/*/solution_ci_*.csv
csv files containing information about the 95% confidence intervals of the fitnesses.
Order of the solutions in these files is consistent.
- hofs/*/solution_features.csv
The features selected by the solutions, a solution for each row. Order of the solutions in these files is consistent.
- hofs/*/solution_fitnesses.csv
The fitnesses of the solutions, a solution for each row. Order of the solutions in these files is consistent.
- hofs/*/solution_std_devs.csv
The fitness standard deviations of the solutions, a solution for each row. Order of the solutions in these files is
consistent.
- hofs/*/validation_registry.json
A JSON file with the numerical values of summary statistics like the cross hypervolume [2] and the Pareto delta [1].
Statistics are saved in this file in order to compute them only once.
- hofs/*/confusion_matrix/
If the setup includes a classification objective, this directory is filled with a csv file representing
the confusion matrix of each solution. Order of the solutions in these files is consistent.

To run the program with a custom dataset, place in the directory work/custom/input a csv file with independent variables
"mrna.csv" and a file with the outcomes "outcome.csv". Each row in the files is a sample, and the order must be
consistent between the two files. The first row is for the header. Each column in the independent variables file is
named with the feature name. The outcomes file can have a column "type" if there is a classification outcome (classes
can be any string), and two columns "Event" and "Time" if there is a survival outcome. Event is 0 (alive) or 1 (dead).
Time is numeric. It is allowed but not necessary to have both a classification and a survival outcome.
A small example dataset is already present in the work/custom/input directory, and an example setup file for running it
is "work/setups/custom_example.ini".

The suggested list of package requirements is in the file requirements.txt.

The R script load_tcga_brca.R creates the mrna.csv and outcome.csv related to the breast TCGA dataset.
Similarly, the script load_tcga_kir.R creates the csv files for the kidney.
In order to work the R scripts require an internet connection. These data files are already present in the work
directory, still the scripts are included for reproducibility.

This project is licensed under the terms of the MIT license.

Bibliography
[1] Luca Cattelani, Vittorio Fortino. "Dual-stage optimizer for systematic overestimation
adjustment applied to multi-objective genetic algorithms for biomarker selection",
arXiv preprint, arXiv:2312.16624 (2023), https://doi.org/10.48550/arXiv.2312.16624
[2] Luca Cattelani, Arindam Ghosh, Teemu Rintala, Vittorio Fortino. "Improving biomarker selection for cancer subtype
classification through multi-objective optimization", TechRxiv (2023), https://doi.org/10.36227/techrxiv.24321154.v2
[3] Luca Cattelani, Vittorio Fortino. "Improved NSGA-II algorithms for multi-objective biomarker discovery",
Bioinformatics, Volume 38, Issue Supplement_2, September 2022, Pages ii20–ii26,
https://doi.org/10.1093/bioinformatics/btac463

