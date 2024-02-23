import cProfile
import pstats

from biodai_cv import run_with_setups

if __name__ == '__main__':
    with cProfile.Profile() as pr:
        run_with_setups()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename="profile.prof")
