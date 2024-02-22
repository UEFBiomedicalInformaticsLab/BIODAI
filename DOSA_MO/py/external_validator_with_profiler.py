import cProfile
import pstats

from external_validator import external_validator

if __name__ == '__main__':
    with cProfile.Profile() as pr:
        external_validator()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename="profile.prof")
