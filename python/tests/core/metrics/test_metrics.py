import numpy as np
import pandas as pd

import whylogs as why
from whylogs.core.metrics.maths import VarianceM2Result, parallel_variance_m2
from whylogs.core.metrics.metrics import DistributionMetric, MetricConfig
from whylogs.core.preprocessing import PreprocessedColumn


def test_distribution_metrics_numpy() -> None:
    dist = DistributionMetric.zero(MetricConfig())
    data = list(range(0, 100))
    arr = np.array(data)
    col = PreprocessedColumn.apply(arr)
    dist.columnar_update(col)

    assert dist.kll.value.get_n() == 100
    assert dist.mean.value == arr.mean()
    assert dist.variance == arr.var()

    distribution_summary = dist.to_summary_dict()
    assert distribution_summary["q_01"] == 1.0
    assert distribution_summary["q_05"] == 5.0
    assert distribution_summary["q_95"] == 95.0
    assert distribution_summary["q_99"] == 99.0


def test_distribution_metrics_series() -> None:
    dist = DistributionMetric.zero(MetricConfig())
    data = pd.Series(list(range(100)))
    col = PreprocessedColumn.apply(data)
    dist.columnar_update(col)

    assert dist.kll.value.get_n() == 100
    assert dist.mean.value == data.mean()
    assert dist.variance == data.var()


def test_distribution_metrics_indexed_series_single_row() -> None:
    dist = DistributionMetric.zero(MetricConfig())
    data = pd.Series(list(range(1)), index=[284])
    col = PreprocessedColumn.apply(data)
    dist.columnar_update(col)

    assert dist.kll.value.get_n() == 1
    assert dist.mean.value == data.mean()


def test_distribution_metrics_list() -> None:
    dist = DistributionMetric.zero(MetricConfig())
    col = PreprocessedColumn()
    data = list(range(0, 100))
    col.list.ints = data
    dist.columnar_update(col)

    assert dist.kll.value.get_n() == 100
    assert dist.mean.value == np.array(data).mean()
    assert dist.variance == np.array(data).var()


def test_distribution_metrics_mixed_np_and_list() -> None:
    dist = DistributionMetric.zero(MetricConfig())
    col = PreprocessedColumn()
    col.list.ints = list(range(0, 50))
    col.numpy.ints = np.array(range(50, 100))
    dist.columnar_update(col)

    assert dist.kll.value.get_n() == 100
    a = np.array(col.list.ints)
    b = col.numpy.ints

    assert dist.mean.value == np.array(np.concatenate([a, b])).mean()

    m2_a = a.var() * (len(a) - 1)
    m2_b = b.var() * (len(b) - 1)
    a_var = VarianceM2Result(n=len(a), mean=a.mean(), m2=m2_a)
    b_var = VarianceM2Result(n=len(b), mean=b.mean(), m2=m2_b)
    overall = parallel_variance_m2(first=a_var, second=b_var)
    assert dist.variance == overall.m2 / (overall.n - 1)


def test_track_single_values_profile_mean() -> None:
    data = list(range(30))
    df = pd.DataFrame(data, columns=["col1"])
    actual_mean = df["col1"].mean()
    actual_stddev = df["col1"].std()
    prof_view_df = why.log(df).profile().view()
    profile_mean1 = prof_view_df.get_column("col1").get_metric("distribution").mean.value
    profile_stddev1 = prof_view_df.get_column("col1").get_metric("distribution").stddev
    for i, d in enumerate(data):
        if i == 0:
            prof_track = why.log(row={"col1": d}).profile()
        else:
            prof_track.track({"col1": d})
    profile_mean2 = prof_track.view().get_column("col1").get_metric("distribution").mean.value
    profile_stddev2 = prof_track.view().get_column("col1").get_metric("distribution").stddev

    assert round(actual_mean, 3) == round(profile_mean1, 3)
    assert round(actual_mean, 3) == round(profile_mean2, 3)
    assert round(actual_stddev, 3) == round(profile_stddev1, 3)
    assert round(actual_stddev, 3) == round(profile_stddev2, 3)


def test_merge_single_values_profile_mean() -> None:
    data = list(range(30))
    df = pd.DataFrame(data, columns=["col1"])
    actual_mean = df["col1"].mean()
    actual_stddev = df["col1"].std()
    prof_view_df = why.log(df).profile().view()
    profile_mean1 = prof_view_df.get_column("col1").get_metric("distribution").mean.value
    profile_stddev1 = prof_view_df.get_column("col1").get_metric("distribution").stddev

    profiles = [why.log(row={"col1": d}).profile().view() for d in data]
    from functools import reduce

    merged_profile_view = reduce((lambda x, y: x.merge(y)), profiles)
    profile_mean2 = merged_profile_view.get_column("col1").get_metric("distribution").mean.value
    profile_stddev2 = merged_profile_view.get_column("col1").get_metric("distribution").stddev

    assert round(actual_mean, 3) == round(profile_mean1, 3)
    assert round(actual_mean, 3) == round(profile_mean2, 3)
    assert round(actual_stddev, 3) == round(profile_stddev1, 3)
    assert round(actual_stddev, 3) == round(profile_stddev2, 3)


def test_merge_two_profiles_mean(lending_club_df) -> None:
    first_df = lending_club_df.head(500)

    second_df = lending_club_df.tail(500)

    actual_mean = lending_club_df["loan_amnt"].mean()
    actual_mean_1 = first_df["loan_amnt"].mean()
    actual_mean_2 = second_df["loan_amnt"].mean()

    first_profile = why.log(first_df).view().get_column("loan_amnt")
    first_profile_mean = first_profile.get_metric("distribution").mean.value
    second_profile = why.log(second_df).view().get_column("loan_amnt")
    second_profile_mean = second_profile.get_metric("distribution").mean.value

    merged_profile = first_profile.merge(second_profile)
    merged_profile_mean = merged_profile.get_metric("distribution").mean.value

    assert round(merged_profile_mean, 3) == round(actual_mean, 3)
    assert round(first_profile_mean, 3) == round(actual_mean_1, 3)
    assert round(second_profile_mean, 3) == round(actual_mean_2, 3)
