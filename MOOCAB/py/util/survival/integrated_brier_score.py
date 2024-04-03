import numpy
from sklearn.utils import check_array, check_consistent_length
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import check_y_survival


def _check_times(test_time, times):
    """Modified starting from the sksurv version."""
    times = check_array(numpy.atleast_1d(times), ensure_2d=False, dtype=test_time.dtype)
    times = numpy.unique(times)
    if times.min() < 0.0:
        raise ValueError("Times must be non-negative.")
    return times


def _check_estimate_2d(estimate, test_time, time_points):
    """Modified starting from the sksurv version."""
    estimate = check_array(estimate, ensure_2d=False, allow_nd=False)
    time_points = _check_times(test_time, time_points)
    check_consistent_length(test_time, estimate)

    if estimate.ndim == 2 and estimate.shape[1] != time_points.shape[0]:
        raise ValueError("expected estimate with {} columns, but got {}".format(
            time_points.shape[0], estimate.shape[1]))

    return estimate, time_points


def brier_score(survival_train, survival_test, estimate, times):
    """Modified starting from the sksurv version.

    The time-dependent Brier score is the mean squared error at time point :math:`t`:

    .. math::

        \\mathrm{BS}^c(t) = \\frac{1}{n} \\sum_{i=1}^n I(y_i \\leq t \\land \\delta_i = 1)
        \\frac{(0 - \\hat{\\pi}(t | \\mathbf{x}_i))^2}{\\hat{G}(y_i)} + I(y_i > t)
        \\frac{(1 - \\hat{\\pi}(t | \\mathbf{x}_i))^2}{\\hat{G}(t)} ,

    where :math:`\\hat{\\pi}(t | \\mathbf{x})` is the predicted probability of
    remaining event-free up to time point :math:`t` for a feature vector :math:`\\mathbf{x}`,
    and :math:`1/\\hat{G}(t)` is a inverse probability of censoring weight, estimated by
    the Kaplan-Meier estimator.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb>` and [1]_ for details.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    estimate : array-like, shape = (n_samples, n_times)
        Estimated risk of experiencing an event for test data at `times`.
        The i-th column must contain the estimated probability of
        remaining event-free up to the i-th time point.

    times : array-like, shape = (n_times,)
        The time points for which to estimate the Brier score.
        Differently from sksurv version, values have no maximum.

    Returns
    -------
    times : array, shape = (n_times,)
        Unique time points at which the brier scores was estimated.

    brier_scores : array , shape = (n_times,)
        Values of the brier score.

    Examples
    --------
    >>> from sksurv.datasets import load_gbsg2
    >>> from sksurv.linear_model import CoxPHSurvivalAnalysis
    >>> from sksurv.metrics import brier_score
    >>> from sksurv.preprocessing import OneHotEncoder

    Load and prepare data.

    >>> X, y = load_gbsg2()
    >>> X.loc[:, "tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)
    >>> Xt = OneHotEncoder().fit_transform(X)

    Fit a Cox model.

    >>> est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)

    Retrieve individual survival functions and get probability
    of remaining event free up to 5 years (=1825 days).

    >>> survs = est.predict_survival_function(Xt)
    >>> preds = [fn(1825) for fn in survs]

    Compute the Brier score at 5 years.

    >>> times, score = brier_score(y, y, preds, 1825)
    >>> print(score)
    [0.20881843]

    See also
    --------
    integrated_brier_score

    References
    ----------
    .. [1] E. Graf, C. Schmoor, W. Sauerbrei, and M. Schumacher,
           "Assessment and comparison of prognostic classification schemes for survival data,"
           Statistics in Medicine, vol. 18, no. 17-18, pp. 2529–2545, 1999.
    """
    test_event, test_time = check_y_survival(survival_test)
    estimate, times = _check_estimate_2d(estimate, test_time, times)
    if estimate.ndim == 1 and times.shape[0] == 1:
        estimate = estimate.reshape(-1, 1)

    # fit IPCW estimator
    cens = CensoringDistributionEstimator().fit(survival_train)
    # calculate inverse probability of censoring weight at current time point t.
    prob_cens_t = cens.predict_proba(times)
    prob_cens_t[prob_cens_t == 0] = numpy.inf
    # calculate inverse probability of censoring weights at observed time point
    prob_cens_y = cens.predict_proba(test_time)
    prob_cens_y[prob_cens_y == 0] = numpy.inf

    # Calculating the brier scores at each time point
    brier_scores = numpy.empty(times.shape[0], dtype=float)
    for i, t in enumerate(times):
        est = estimate[:, i]
        is_case = (test_time <= t) & test_event
        is_control = test_time > t

        brier_scores[i] = numpy.mean(numpy.square(est) * is_case.astype(int) / prob_cens_y
                                     + numpy.square(1.0 - est) * is_control.astype(int) / prob_cens_t[i])

    return times, brier_scores


def integrated_brier_score(survival_train, survival_test, estimate, times):
    """Modified starting from integrated_brier_score of sksurv.

    The Integrated Brier Score (IBS) provides an overall calculation of
    the model performance at all available times :math:`t_1 \\leq t \\leq t_\\text{max}`.

    The integrated time-dependent Brier score over the interval
    :math:`[t_1; t_\\text{max}]` is defined as

    .. math::

        \\mathrm{IBS} = \\int_{t_1}^{t_\\text{max}} \\mathrm{BS}^c(t) d w(t)

    where the weighting function is :math:`w(t) = t / t_\\text{max}`.
    The integral is estimated via the trapezoidal rule.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb>`
    and [1]_ for further details.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    estimate : array-like, shape = (n_samples, n_times)
        Estimated risk of experiencing an event for test data at `times`.
        The i-th column must contain the estimated probability of
        remaining event-free up to the i-th time point.

    times : array-like, shape = (n_times,)
        The time points for which to estimate the Brier score.
        Differently from sksurv version, values have no maximum.
        They must be sorted and unique.

    Returns
    -------
    ibs : float
        The integrated Brier score.

    Examples
    --------
    >>> import numpy
    >>> from sksurv.datasets import load_gbsg2
    >>> from sksurv.linear_model import CoxPHSurvivalAnalysis
    >>> from sksurv.metrics import integrated_brier_score
    >>> from sksurv.preprocessing import OneHotEncoder

    Load and prepare data.

    >>> X, y = load_gbsg2()
    >>> X.loc[:, "tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)
    >>> Xt = OneHotEncoder().fit_transform(X)

    Fit a Cox model.

    >>> est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)

    Retrieve individual survival functions and get probability
    of remaining event free from 1 year to 5 years (=1825 days).

    >>> survs = est.predict_survival_function(Xt)
    >>> times = numpy.arange(365, 1826)
    >>> preds = numpy.asarray([[fn(t) for t in times] for fn in survs])

    Compute the integrated Brier score from 1 to 5 years.

    >>> score = integrated_brier_score(y, y, preds, times)
    >>> print(score)
    0.1815853064627424

    See also
    --------
    brier_score

    References
    ----------
    .. [1] E. Graf, C. Schmoor, W. Sauerbrei, and M. Schumacher,
           "Assessment and comparison of prognostic classification schemes for survival data,"
           Statistics in Medicine, vol. 18, no. 17-18, pp. 2529–2545, 1999.
    """
    # Computing the brier scores
    times, brier_scores = brier_score(survival_train, survival_test, estimate, times)

    if times.shape[0] < 2:
        raise ValueError("At least two time points must be given")

    # Computing the IBS
    ibs_value = numpy.trapz(brier_scores, times) / (times[-1] - times[0])

    return ibs_value
