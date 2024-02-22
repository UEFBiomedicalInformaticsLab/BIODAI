from collections.abc import Iterable, Sequence
from math import sqrt

from util.hyperbox.hyperbox import Interval, ConcreteInterval
from util.math.summer import KahanSummer


def combined_variance(means: Iterable[float], variances: Iterable[float]) -> float:
    """Assumption: means and variances are from independent measurements of the same random variable."""
    meta_mean = KahanSummer.mean(means)
    res = KahanSummer()
    n = 0
    for m, v in zip(means, variances):
        n += 1
        res.add(v)
        res.add((m-meta_mean)**2)
    return res.get_sum() / float(n)


def variance_of_uncorrelated_sum(variances: Iterable[float]) -> float:
    """Assumption: the random variables are uncorrelated. In this case the variance of the sum is the sum
    of the variances."""
    return KahanSummer().sum(variances)


def variance_of_uncorrelated_mean(variances: Sequence[float]) -> float:
    """Assumption: the random variables are uncorrelated."""
    return KahanSummer().sum(variances) / (len(variances)**2)


def combined_std_dev(means: Iterable[float], std_devs: Iterable[float]) -> float:
    """Assumption: means and variances are from independent measurements of the same random variable."""
    variances = [sd**2 for sd in std_devs]
    combined_var = combined_variance(means=means, variances=variances)
    return sqrt(combined_var)


def std_dev_of_uncorrelated_sum(std_devs: Iterable[float]) -> float:
    """Assumption: the random variables are uncorrelated. In this case the variance of the sum is the sum
    of the variances."""
    variances = [sd ** 2 for sd in std_devs]
    sum_var = variance_of_uncorrelated_sum(variances=variances)
    return sqrt(sum_var)


def std_dev_of_uncorrelated_mean(std_devs: Iterable[float]) -> float:
    """Assumption: the random variables are uncorrelated."""
    variances = [sd ** 2 for sd in std_devs]
    sum_var = variance_of_uncorrelated_sum(variances=variances)
    return sqrt(sum_var) / len(variances)


def confidence_interval_of_uncorrelated_sum(confidence_intervals: Sequence[Interval]) -> Interval:
    """Assumption: the passed confidence intervals are computed for uncorrelated random variables.
    The intervals must be of the same aperture, e.g. all 95%.
    It is assumed that the confidence intervals are from normal distributions.
    Notice that by central limit theorem if each confidence interval is computed with bootstrap,
    as the number of repetitions increase the shape of the distribution of
    the quality metric approaches the Gaussian.
    The returned confidence interval is the CI for the sum of the random variables
    of which the input CIs are measured."""
    tot = KahanSummer.sum([i.mid_pos() for i in confidence_intervals])
    radius = sqrt(KahanSummer.sum([(i.length())**2 for i in confidence_intervals])) / 2.0
    return ConcreteInterval(a=tot-radius, b=tot+radius)


def confidence_interval_of_uncorrelated_mean(confidence_intervals: Sequence[Interval]) -> Interval:
    """Assumption: the passed confidence intervals are computed for uncorrelated random variables.
    The intervals must be of the same aperture, e.g. all 95%.
    It is assumed that the confidence intervals are from normal distributions.
    Notice that by central limit theorem if each confidence interval is computed with bootstrap,
    as the number of repetitions increase the shape of the distribution of
    the quality metric approaches the Gaussian.
    The returned confidence interval is the CI for the mean of the random variables
    of which the input CIs are measured."""
    mean = KahanSummer.mean([i.mid_pos() for i in confidence_intervals])
    radius = sqrt(KahanSummer.sum([(i.length())**2 for i in confidence_intervals])) / (2.0 * len(confidence_intervals))
    return ConcreteInterval(a=mean-radius, b=mean+radius)
