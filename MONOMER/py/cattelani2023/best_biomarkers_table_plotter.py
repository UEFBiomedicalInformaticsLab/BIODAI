from collections.abc import Sequence

from pandas import DataFrame

from cattelani2023.cattelani2023_utils import cattelani2023_external_hofs, CATTELANI2023_DIR
from plots.best_solutions_printer import best_solutions_for_dataset, solution_str
from saved_solutions.solution_from_algorithm import SolutionFromAlgorithm

MAX1 = 10
MAX2 = 20
MAX3 = 30


def best_solution(
        solutions: Sequence[SolutionFromAlgorithm], min_features: int, max_features: int) -> SolutionFromAlgorithm:
    """Chooses on the first fitness value."""
    best = None
    best_fit = -1
    for s in solutions:
        s_features = s.num_features()
        if min_features <= s_features <= max_features:
            fit = s.get_test_fitness().values[0]
            if fit > best_fit:
                best = s
                best_fit = fit
    return best


if __name__ == '__main__':
    BEST_BIO_FILE_NAME = "best_biomarkers.csv"
    res = DataFrame(index=range(4), columns=range(3), dtype=str)
    res.columns = ["0 - 10", "11 - 20", "21 - 30"]
    res.index = ["Breast", "Kidney", "Lung", "Ovary"]
    for row, ext in enumerate(cattelani2023_external_hofs()):
        hofers = best_solutions_for_dataset(hofs=ext)
        best1 = best_solution(hofers, min_features=0, max_features=MAX1)
        best2 = best_solution(hofers, min_features=MAX1+1, max_features=MAX2)
        best3 = best_solution(hofers, min_features=MAX2+1, max_features=MAX3)
        res.iloc[row, 0] = solution_str(best1)
        res.iloc[row, 1] = solution_str(best2)
        res.iloc[row, 2] = solution_str(best3)
    res.to_csv(CATTELANI2023_DIR + "/" + "best_biomarkers_table.csv")
