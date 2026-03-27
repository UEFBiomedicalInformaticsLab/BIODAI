from individual.num_features import NumFeatures, UniformNumFeatures, BinomialNumFeatures, BinomialFromUniformNumFeatures
from setup.allowed_names import INITIAL_FEATURES_UNIFORM_NAME, INITIAL_FEATURES_BINOMIAL_NAME, \
    INITIAL_FEATURES_BINOMIAL_FROM_UNIFORM_NAME


def parse_initial_features(
        initial_features_strategy_str: str,
        initial_features_min: int,
        initial_features_max: int) -> NumFeatures:
    if initial_features_strategy_str == INITIAL_FEATURES_UNIFORM_NAME:
        return UniformNumFeatures(min_num_features=initial_features_min, max_num_features=initial_features_max)
    if initial_features_strategy_str == INITIAL_FEATURES_BINOMIAL_NAME:
        print("Creating BinomialNumFeatures with default mean.")
        return BinomialNumFeatures()
    if initial_features_strategy_str == INITIAL_FEATURES_BINOMIAL_FROM_UNIFORM_NAME:
        return BinomialFromUniformNumFeatures(
            min_num_features=initial_features_min, max_num_features=initial_features_max)
    raise ValueError("Unknown initial features strategy.")
