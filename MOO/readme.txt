Publicly available source code and data related to
Luca Cattelani, Vittorio Fortino, Improved NSGA-II algorithms for multi-objective biomarker discovery, Bioinformatics,
Volume 38, Issue Supplement_2, September 2022, Pages ii20–ii26, https://doi.org/10.1093/bioinformatics/btac463

The main program is in Python, we include also R scripts that were used to prepare the TCGA breast dataset.

The R script load_tcga_brca.R creates the mrna.csv and outcome.csv files that are needed for the tests.
In order to do that it requires an internet connection and the file pheno.csv that is already included in this
repository.

All the data files needed to run the tests are included in this repository, except for the mrna data because
it is too large for GitHub. All the results, including plots, are included in this repository,
but can also be generated again by a user launching the Python programs.

The main Python programs get in input an INI file with the configuration of a test. All the INI file used to produce
the paper results are in the directory work/setups.
A configuration file starts with [MVMOO_SETUP].
The most important parameters are the following.

- dataset. The name of the dataset to be used. We use "brca" in the included tests.
- mvmo_algorithm. The name of the main algorithm. It can be "lasso_mo", or "classic_ga" for NSGA2 and the new variants.
- objectives. A string that specifies the objectives and if required also the inner model. For balanced accuracy and
leanness with naive Bayes as inner model it is "[["bal_acc", "naive_bayes"], "leanness"]". The inner model can be
naive_bayes", "RF" (random forest), or "logistic". The inner model is ignored by lasso_mo.
- use_big_defaults. Boolean parameter, the default is "false" and some parameters are set for a short test run.
When true these parameters are set for a long serious run. This must be set to true to reproduce our results.
- external_dataset. The name of the external dataset. We use "swedish" (SCAN-B cohort) in our tests. This parameter
is ignored when running cross-validation.
- cross_validation. If true and if not running an external validation, the cross-validation is performed, and the
results saved.
- final_optimization. If true and if not running an external validation, the optimization on the whole dataset is
performed, and the results saved.
- pop. The size of the population for GA based algorithms.
- generations. A list of integers to support future extensions, we use lists of only one integer in these tests.
The value in the list is the number of generations used by the GA-based algorithms.
- initial_features_strategy. Strategy to extract the number of features when initializing a solution.
We always use "uniform" in our tests, so that the number of features is extracted with a uniform distribution.
Other two parameters are used to set the minimum and maximum number of features in an initial solution.
- initial_features_min and initial_features_max. Two numbers to specify the minimum and maximum number of features in an
initial solution.
- sorting_strategy. The sorting strategy to use before selection and tournament. It can be "crowding_distance_full"
for NSGA2 implied sorting or "crowding_distance_clone_index" to use the clone index as primary sorting criteria.
- use_clone_repurposing. A Boolean. If true clone repurposing is used.
- logistic_max_iter. The maximum number of iterations used by the logistic regression inner model. Ignored when using
another inner model.
- feature_importance_categorical. The strategy used to compute the feature importance for the GA-based algorithms.
With the default of "none" a uniform feature importance is used. With "lasso" the LASSO feature importance is used.
- bitlist_mutation_operator. With the default of "flip" a bit-flip operator is used. With "symm" the symmetric mutation
is used instead.

The Python script py/run_with_setups.py is used to run the k-fold cross-validation and the final optimization
(optimization on the whole dataset). It gets in input an INI setup file. An example of run from command line
(from inside the working directory "work") is
python ../py/run_with_setups.py setups/brca/test_small.ini

The Python script py/external_validator.py runs an external validation. It gets in input an INI setup file
and works in an analogous way as run_with_setups.py.

In py/plots there are scripts for creating summary plots. It is sufficient to run them without any parameter, and they
will search the work directory for the necessary test results and create the plots. The working directory must be the
directory "work". The scripts are
- eccb_fig1.py
- folds_scatter_plots_runner.py
- multi_front_plotter.py
- multi_objective_pairs_plotter.py
- subplots_runner.py

This project is licensed under the terms of the MIT license.
