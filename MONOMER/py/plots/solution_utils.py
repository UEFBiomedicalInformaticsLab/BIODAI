import os
from typing import Sequence

from plots.saved_hof import SavedHoF
from saved_solutions.saved_solution import SavedSolution
from saved_solutions.solution_from_algorithm import SolutionFromAlgorithm
from saved_solutions.solutions_from_files import final_solutions_from_files


def solutions_and_hof_names(hofs: Sequence[SavedHoF]) -> (list[Sequence[SavedSolution]], list[str]):
    solutions = []
    hof_names = []
    for alg_hofs in hofs:
        h_path = alg_hofs.path()
        if os.path.isdir(h_path):
            solutions.append(final_solutions_from_files(hof_dir=h_path))
            hof_names.append(alg_hofs.name())
        else:
            print("path is not directory: " + str(h_path))
    return solutions, hof_names


def solutions_from_algorithms(hofs: Sequence[SavedHoF]) -> list[SolutionFromAlgorithm]:
    all_solutions, hof_names = solutions_and_hof_names(hofs=hofs)
    res = []
    for solutions, hof_name in zip(all_solutions, hof_names):
        for solution in solutions:
            res.append(SolutionFromAlgorithm(solution=solution, algorithm_name=hof_name))
    return res
