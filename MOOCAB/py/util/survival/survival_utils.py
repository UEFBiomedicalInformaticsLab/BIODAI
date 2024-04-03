import numpy as np
from pandas import DataFrame
from sklearn.utils import check_consistent_length

from util.survival.integrated_brier_score import integrated_brier_score

SURVIVAL_DURATION_STR = 'duration'
SURVIVAL_EVENT_STR = 'event'


def survival_times(survival_data: DataFrame) -> list[float]:
    return survival_data[SURVIVAL_DURATION_STR].values.tolist()


def survival_events(survival_data: DataFrame) -> list[bool]:
    if isinstance(survival_data, DataFrame):
        return survival_data[SURVIVAL_EVENT_STR].values.tolist()
    else:
        raise ValueError("Object passed is not a DataFrame. Passed object: " + str(survival_data))


def surv_from_arrays(event, time, name_event=None, name_time=None):
    """Create structured array. From sksurv, but this version does not complain if
    events are all 0 or all 1.

    Parameters
    ----------
    event : array-like
        Event indicator. A boolean array or array with values 0/1.
    time : array-like
        Observed time.
    name_event : str|None
        Name of event, optional, default: 'event'
    name_time : str|None
        Name of observed time, optional, default: 'time'

    Returns
    -------
    y : np.array
        Structured array with two fields.
    """
    name_event = name_event or "event"
    name_time = name_time or "time"
    if name_time == name_event:
        raise ValueError("name_time must be different from name_event")

    time = np.asanyarray(time, dtype=float)
    y = np.empty(time.shape[0], dtype=[(name_event, bool), (name_time, float)])
    y[name_time] = time

    event = np.asanyarray(event)
    check_consistent_length(time, event)

    if np.issubdtype(event.dtype, np.bool_):
        y[name_event] = event
    else:
        events = np.unique(event)
        if len(events) > 2:
            raise ValueError("event indicator must be binary")
        accepted_events = np.array([0, 1], dtype=events.dtype)
        for e in events:
            if e not in accepted_events:
                raise ValueError("non-boolean event indicator must contain 0 and 1 only")
        y[name_event] = event.astype(bool)

    return y


def survival_df_to_sksurv(survival_df: DataFrame):
    try:
        return surv_from_arrays(
            event=survival_events(survival_data=survival_df), time=survival_times(survival_data=survival_df))
    except ValueError as e:
        raise ValueError("survival_df:\n" + str(survival_df) + "\n" + "Original cause:\n" + str(e))


def integrated_brier_score_from_df(
        surv_for_censoring_df: DataFrame, surv_test_df: DataFrame, estimate: DataFrame, times: [float]) -> float:
    """Censoring survival data is used to train the censoring estimator, more data the better.
    It is suggested to include all the available data, including the test data.
    times is the times at which to compute the Brier score, they must be sorted in increasing order and unique.
    estimate[i,j] is the model estimates of event up to the
    corresponding time[j], for test individual i."""
    train_array = survival_df_to_sksurv(survival_df=surv_for_censoring_df)
    test_array = survival_df_to_sksurv(survival_df=surv_test_df)
    return integrated_brier_score(
        survival_train=train_array, survival_test=test_array, estimate=estimate, times=times)
