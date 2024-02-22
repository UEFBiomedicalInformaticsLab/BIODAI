import pandas
from pandas import DataFrame

from folds_creator.mo_class_assigner.mo_class_assigner import MOClassAssigner
from folds_creator.mo_class_assigner.strata import Strata, integrate_strata
from input_data.input_data import InputData
from input_data.outcome import OutcomeType
from util.survival.survival_utils import survival_times, survival_events
from util.printer.printer import Printer, NullPrinter
from util.utils import PlannedUnreachableCodeError


DEFAULT_NUM_TIME_STRATA = 2


def time_label(time: float, categories) -> int:
    last_category = len(categories) - 1
    lower = categories[0].left
    upper = categories[last_category].right
    if time <= lower:
        return 0
    if time >= upper:
        return last_category
    for i, c in enumerate(categories):
        if time in c:
            return i
    raise PlannedUnreachableCodeError()


def survival_event_str(event: bool) -> str:
    if event:
        return "deceased"
    else:
        return "censored"


def survival_outcome_str(time_stratum: int, event: bool) -> str:
    return "time " + str(time_stratum) + " " + survival_event_str(event=event)


def event_strata(survival_data: DataFrame, min_stratum_size: int = 1) -> Strata:
    events = survival_events(survival_data)
    return Strata.create_from_names([survival_event_str(e) for e in events], min_size=min_stratum_size)


def time_strata(
        survival_data: DataFrame, n_time_strata: int = DEFAULT_NUM_TIME_STRATA, min_stratum_size: int = 1) -> Strata:
    times = survival_times(survival_data)
    events = survival_events(survival_data)
    n_individuals = len(times)
    times_with_event = []
    for i in range(n_individuals):
        if events[i]:
            times_with_event.append(times[i])
    cuts = pandas.qcut(x=times_with_event, q=n_time_strata)
    categories = cuts.categories
    time_labels = [time_label(t, categories) for t in times]
    return Strata.create_from_ids(time_labels, min_size=min_stratum_size)


def survival_strata(data: DataFrame, n_time_strata: int = DEFAULT_NUM_TIME_STRATA, min_stratum_size: int = 1) -> Strata:
    event_s = event_strata(survival_data=data, min_stratum_size=min_stratum_size)
    time_s = time_strata(survival_data=data, n_time_strata=n_time_strata, min_stratum_size=min_stratum_size)
    return integrate_strata(strata1=event_s, strata2=time_s, min_size=min_stratum_size)


class FullClassAssigner(MOClassAssigner):

    def assign_classes(self, data: InputData, printer: Printer = NullPrinter(), min_stratum_size: int = 1) -> Strata:
        printer.title_print("Assigning strata for multi-outcome data " + data.name())
        so_assignments = []
        for o in data.outcomes():
            o_type = o.type()
            printer.print("Assigning strata for outcome named " + o.name() + " of type " + str(o_type.name))
            if o_type == OutcomeType.categorical or o_type == OutcomeType.survival:
                if o.type() == OutcomeType.categorical:
                    strata = Strata.create_from_names(o.data().iloc[:, 0], min_size=min_stratum_size)
                else:  # Survival
                    strata = survival_strata(data=o.data(), min_stratum_size=min_stratum_size)
                printer.print(str(strata))
                so_assignments.append(strata)
            else:
                printer.print("Unsupported outcome type.")
        n_used_outcomes = len(so_assignments)
        if n_used_outcomes == 0:
            printer.print("No supported outcomes for stratification, putting everything in the same stratum.")
            return Strata.create_one_stratum(n_samples=data.n_samples())
        else:
            res = so_assignments[0]
            if n_used_outcomes > 1:
                for i in range(1, n_used_outcomes):
                    res = integrate_strata(res, so_assignments[i], min_size=min_stratum_size)
                printer.print("Strata from each outcome has been integrated.\n" + str(res))
            return res
