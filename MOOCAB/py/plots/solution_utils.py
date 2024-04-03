import os
from typing import Sequence, Optional

from plots.saved_hof import SavedHoF
from saved_solutions.saved_solution import SavedSolution
from saved_solutions.solution_from_algorithm import SolutionFromAlgorithm
from saved_solutions.solutions_from_files import final_solutions_from_files, solutions_from_files
from util.utils import names_by_differences


def solutions_and_hof_names(
        hofs: Sequence[SavedHoF], verbose: bool = False) -> tuple[list[Sequence[SavedSolution]], Sequence[str]]:
    """Returns the final solutions. Hof names are shortened removing the parts that are always the same."""
    solutions = []
    hof_name_parts = []
    for alg_hofs in hofs:
        h_path = alg_hofs.path()
        if os.path.isdir(h_path):
            solutions.append(final_solutions_from_files(hof_dir=h_path))
            hof_name_parts.append(alg_hofs.name_parts())
        else:
            if verbose:
                print("path is not directory: " + str(h_path))
    return solutions, names_by_differences(object_features=hof_name_parts)


def obj_nicks_from_hofs(hofs: Sequence[SavedHoF]) -> Optional[Sequence[str]]:
    for h in hofs:
        if h.has_obj_nicks():
            return h.obj_nicks()
    return None


def solutions_from_algorithms(hofs: Sequence[SavedHoF]) -> list[SolutionFromAlgorithm]:
    all_solutions, hof_names = solutions_and_hof_names(hofs=hofs)
    res = []
    for solutions, hof_name in zip(all_solutions, hof_names):
        for solution in solutions:
            res.append(SolutionFromAlgorithm(solution=solution, algorithm_name=hof_name))
    return res


def solutions_all_folds_and_hof_names(
        hofs: Sequence[SavedHoF], verbose: bool = False) -> tuple[list[Sequence[Sequence[SavedSolution]]], list[str]]:
    """Returns the cv folds solutions."""
    solutions = []
    hof_names = []
    for alg_hofs in hofs:
        h_path = alg_hofs.path()
        if os.path.isdir(h_path):
            solutions.append(solutions_from_files(hof_dir=h_path))
            hof_names.append(alg_hofs.name())
        else:
            if verbose:
                print("path is not directory: " + str(h_path))
    return solutions, hof_names
