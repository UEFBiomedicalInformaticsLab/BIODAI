from individual.fitness.high_best_fitness import HighBestFitness


class PeculiarFitness(HighBestFitness):
    __social_space: float
    __peculiarity: float
    crowding_dist: float  # Not private for compatibility with DEAP

    def __init__(self, n_objectives: int, values=()):
        super().__init__(n_objectives=n_objectives, values=values)
        self.__peculiarity = None
        self.crowding_dist = None
        self.__social_space = None

    def set_crowding_distance(self, crowding_distance: float):
        self.crowding_dist = crowding_distance

    def get_crowding_distance(self):
        return self.crowding_dist

    def set_peculiarity(self, peculiarity: float):
        self.__peculiarity = peculiarity

    def get_peculiarity(self):
        return self.__peculiarity

    def set_social_space(self, social_space: float):
        self.__social_space = social_space

    def get_social_space(self):
        return self.__social_space