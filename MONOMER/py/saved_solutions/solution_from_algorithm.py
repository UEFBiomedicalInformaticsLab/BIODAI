from individual.fit import Fit
from individual.fitness.high_best_fitness import HighBestFitness
from saved_solutions.saved_solution import SavedSolution


class SolutionFromAlgorithm(Fit):
    __solution: SavedSolution
    __algorithm: str

    def __init__(self, solution: SavedSolution, algorithm_name: str):
        self.__solution = solution
        self.__algorithm = algorithm_name

    def has_fitness(self) -> bool:
        return self.__solution.has_fitnesses()

    def get_test_fitness(self) -> HighBestFitness:
        return self.__solution.get_test_fitness()

    def solution(self) -> SavedSolution:
        return self.__solution

    def num_features(self) -> int:
        return self.solution().num_features()

    def algorithm_name(self) -> str:
        return self.__algorithm

    def __str__(self) -> str:
        return str(self.__algorithm) + " " + str(self.__solution)
