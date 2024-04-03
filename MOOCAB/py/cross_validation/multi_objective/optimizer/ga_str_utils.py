from collections.abc import Sequence

from ga_components.bitlist_mutation import BitlistMutation
from ga_components.sorter.sorting_strategy import SortingStrategy
from util.utils import str_paste

NICK_SEPARATOR = "_"
NAME_SEPARATOR = ", "


def nick_paste(parts: Sequence[str]) -> str:
    return str_paste(parts=parts, separator=NICK_SEPARATOR)


def name_paste(parts: Sequence[str]) -> str:
    return str_paste(parts=parts, separator=NAME_SEPARATOR)


def pop_nick(pop_size: int) -> str:
    return "pop" + str(pop_size)


def pop_name(pop_size: int) -> str:
    return "pop " + str(pop_size)


def gen_nick(n_gen: int) -> str:
    return "gen" + str(n_gen)


def gen_name(n_gen: int) -> str:
    return "gen " + str(n_gen)


def sorting_strategy_nick_part(sorting_strategy: SortingStrategy, use_clone_repurposing: bool) -> str:
    return sorting_strategy.nick() + ("CR" if use_clone_repurposing else "")


def sorting_strategy_name_part(sorting_strategy: SortingStrategy, use_clone_repurposing: bool) -> str:
    return "sort " + sorting_strategy.nick() + ("CR" if use_clone_repurposing else "")


def mating_prob_nick(mating_prob: float) -> str:
    return "c" + str(round(mating_prob, 2))


def mating_prob_name(mating_prob: float) -> str:
    return "crossover " + str(round(mating_prob, 2))


def mutation_nick_part(mutation: BitlistMutation, mutation_frequency: float) -> str:
    return "m" + str(round(mutation_frequency, 2)) + mutation.nick()


def mutation_name_part(mutation: BitlistMutation, mutation_frequency: float) -> str:
    return "mutation " + str(round(mutation_frequency, 2)) + " " + mutation.name()
