# BIODAI – BIOmarker Discovery through Artificial Intelligence

This repository contains a collection of research projects developed within the University of Eastern Finland (UEF) Biomedical Informatics Research Group (FortinoLab).
Each subdirectory corresponds to a standalone research project, including source code used for the experimental parts of published or ongoing scientific works.

## Overview

The goal of this repository is to provide:
- Reproducible implementations of methods for biomarker discovery and machine learning in biomedical data
- Algorithms for multi-objective optimization, feature selection, and multi-omic integration
- Experimental workflows used in peer-reviewed publications and preprints

The projects focus on omics-based biomarker discovery, where high-dimensional molecular data and limited sample sizes pose significant challenges.

Each subdirectory represents an independent project. To understand a specific project, refer to the README or documentation inside its subdirectory.

## Research Themes

### Multi-objective feature selection

- Optimization of predictive performance vs. model complexity
- Identification of Pareto-optimal biomarker panels


### Genetic algorithms for biomarker discovery

- Wrapper-based feature selection using machine learning models
- Efficient search of high-dimensional feature spaces
- Use of evolutionary algorithms such as NSGA variants [1,3,4]
- Robust evaluation with cross-validation and external datasets [2]

### Multi-omics data integration

- Integration of heterogeneous data (e.g. clinical, mRNA, miRNA)
- Multi-view optimization frameworks
- Exploration of cross-modal interactions [5]

### Addressing overestimation in model selection

- Methods to reduce performance overestimation
- Improved selection of models during optimization
- DOSA-MO algorithm for adjusting fitness estimates [4]

## Related Publications

The repository contains implementations supporting the experimental results of several works, including:

- *A comprehensive evaluation framework for benchmarking multi-objective feature selection in omics-based biomarker discovery.*
  Provides benchmarking methods and evaluation metrics for multi-objective optimization such as Cross Hypervolume (CHV) [2].

- *Dual-stage optimizer for systematic overestimation adjustment applied to multi-objective genetic algorithms for biomarker selection.*
  Introduces DOSA-MO, a framework to reduce overestimation during optimization [4].

- *Genetic algorithms for multi-omic feature selection: a comparative study in cancer survival analysis.*
  Introduces multi-view and multi-objective optimization strategies (algorithm Sweeping*) [5].

## Code and Data

This repository is research-oriented. Code is organized per project.
Please refer to the individual project documentation for setup and usage instructions.

Experiments rely on publicly available datasets, such as The Cancer Genome Atlas (TCGA).
Preprocessed datasets or scripts may be included in subprojects.

## Key Concepts

- Multi-objective optimization (MO):
Simultaneous optimization of multiple conflicting objectives (e.g. accuracy vs. feature count)

- Pareto front:
Set of optimal trade-offs between objectives

- Wrapper feature selection:
Evaluating feature subsets using machine learning models during optimization

- Overestimation:
Performance inflation due to selecting the best models among many candidates [4]

## Authors
Developed by members of the UEF Biomedical Informatics Research Group:
- Luca Cattelani
- Vittorio Fortino
- Collaborators and contributors (see individual projects)

## License

These projects are licensed under the terms of the MIT license.

## Bibliography

[1] Luca Cattelani, Vittorio Fortino. Improved NSGA-II algorithms for multi-objective biomarker discovery. Bioinformatics, Volume 38, Issue Supplement_2, September 2022, Pages ii20–ii26. URL https://doi.org/10.1093/bioinformatics/btac463

[2] Luca Cattelani, Arindam Ghosh, Teemu Rintala, Vittorio Fortino. A Comprehensive Evaluation Framework for Benchmarking Multi-Objective Feature Selection in Omics-Based Biomarker Discovery. In IEEE/ACM Transactions on Computational Biology and Bioinformatics, vol. 21, no. 6, pp. 2432-2446, Nov.-Dec. 2024. URL https://doi.org/10.1109/TCBB.2024.3480150

[3] Luca Cattelani, Vittorio Fortino.
Triple and quadruple optimization for feature selection in cancer biomarker discovery.
Journal of Biomedical Informatics,
Volume 159, 2024, 104736, ISSN 1532-0464. URL https://doi.org/10.1016/j.jbi.2024.104736

[4] Luca Cattelani, Vittorio Fortino. Dual-stage optimizer for systematic overestimation adjustment applied to multi-objective genetic algorithms for biomarker selection. Briefings in Bioinformatics, Volume 26, Issue 1, January 2025, bbae674. URL https://doi.org/10.1093/bib/bbae674

[5] Luca Cattelani, Vittorio Fortino. “Genetic algorithms for multi-omic feature selection: A comparative study in cancer survival analysis,” arXiv preprint, arXiv:2604.00065, 2026. doi: 10.48550/arXiv.2604.00065. URL https://arxiv.org/abs/2604.00065
