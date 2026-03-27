from typing import Optional

from util.hyperbox.hyperbox import Interval


class ValidateOnePredictorRes:
    objective_on_test: Optional[float]
    objective_on_test_ci: Optional[Interval]
    objective_on_train: Optional[float]
    objective_on_train_ci: Optional[Interval]

    def __init__(self):
        self.objective_on_test = None
        self.objective_on_test_ci = None
        self.objective_on_train = None
        self.objective_on_train_ci = None
