import cProfile
import pstats

from consts import PROFILE_FILE
from plot_all_from_batteries import COMMANDS, plot_all_from_commands

if __name__ == '__main__':
    with cProfile.Profile() as pr:
        plot_all_from_commands(commands=COMMANDS)
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename=PROFILE_FILE)
