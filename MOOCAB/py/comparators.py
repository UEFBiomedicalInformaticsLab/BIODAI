from collections import Iterable
from functools import cmp_to_key

from individual.fit_individual import FitIndividual
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from util.named import Named
from util.math.summer import KahanSummer


class Comparator(Named):

    def compare(self, x: PeculiarIndividualByListlike, y: PeculiarIndividualByListlike) -> int:
        raise NotImplementedError()

    def to_key(self):
        return cmp_to_key(mycmp=self.compare)


class CompositeComparator(Comparator):
    """Comparators are applied in order until one gives non-zero"""

    __comparators: Iterable[Comparator]

    def __init__(self, comparators: Iterable[Comparator]):
        self.__comparators = comparators

    def compare(self, x, y):
        for c in self.__comparators:
            c_res = c.compare(x, y)
            if c_res != 0:
                return c_res
        return 0

    def name(self):
        res = "composite comparator ["
        first = True
        for c in self.__comparators:
            if not first:
                res += ", "
            res += c.name()
            first = False
        res += "]"
        return res


class ComparatorDomination(Comparator):

    @staticmethod
    def compare(x: PeculiarIndividualByListlike, y: PeculiarIndividualByListlike):
        x_fit = x.fitness
        y_fit = y.fitness
        if x_fit.dominates(other=y_fit):
            return -1
        elif y_fit.dominates(other=x_fit):
            return 1
        else:
            return 0

    def name(self):
        return "comparator on domination"


class ComparatorOnCrowdingDistance(Comparator):

    @staticmethod
    def compare(x: PeculiarIndividualByListlike, y: PeculiarIndividualByListlike):
        x_cd = x.get_crowding_distance()
        y_cd = y.get_crowding_distance()
        if x_cd > y_cd:
            return -1
        elif y_cd > x_cd:
            return 1
        else:
            return 0

    def name(self):
        return "comparator on crowding distance"


class ComparatorOnSocialSpace(Comparator):

    @staticmethod
    def compare(x: PeculiarIndividualByListlike, y: PeculiarIndividualByListlike):
        x_cd = x.get_social_space()
        y_cd = y.get_social_space()
        if x_cd > y_cd:
            return -1
        elif y_cd > x_cd:
            return 1
        else:
            return 0

    def name(self):
        return "comparator on social space"


class ComparatorOnInterest(Comparator):

    @staticmethod
    def compare(x: PeculiarIndividualByListlike, y: PeculiarIndividualByListlike):
        x_cd = x.get_crowding_distance()
        y_cd = y.get_crowding_distance()
        x_p = x.get_peculiarity()
        y_p = y.get_peculiarity()
        x_combined = x_cd * x_p  # TODO: Would probably be better to have normalized values and sum them
        y_combined = y_cd * y_p
        if x_combined > y_combined:
            return -1
        elif y_combined > x_combined:
            return 1
        else:
            return 0

    def name(self):
        return "comparator on interest"


class ComparatorOnDominationAndCrowding(Comparator):

    __inner = CompositeComparator(comparators=(ComparatorDomination(), ComparatorOnCrowdingDistance()))

    @classmethod
    def compare(cls, x: PeculiarIndividualByListlike, y: PeculiarIndividualByListlike):
        return cls.__inner.compare(x, y)

    def name(self):
        return self.__inner.name()


class ComparatorOnDominationAndSocialSpace(Comparator):

    __inner = CompositeComparator(comparators=(ComparatorDomination(), ComparatorOnSocialSpace()))

    @classmethod
    def compare(cls, x: PeculiarIndividualByListlike, y: PeculiarIndividualByListlike):
        return cls.__inner.compare(x, y)

    def name(self):
        return self.__inner.name()


class ComparatorOnDominationAndInterest(Comparator):

    __inner = CompositeComparator(comparators=(ComparatorDomination(), ComparatorOnInterest()))

    @classmethod
    def compare(cls, x: PeculiarIndividualByListlike, y: PeculiarIndividualByListlike):
        return cls.__inner.compare(x, y)

    def name(self):
        return self.__inner.name()


class ComparatorOnSum(Comparator):

    @staticmethod
    def compare(x: FitIndividual, y: FitIndividual) -> int:
        x_fit = x.fitness.values
        y_fit = y.fitness.values
        sum_x = KahanSummer.sum(x_fit)
        sum_y = KahanSummer.sum(y_fit)
        if sum_x > sum_y:
            return -1
        elif sum_y > sum_x:
            return 1
        else:
            return 0

    def name(self):
        return "comparator on sum"
