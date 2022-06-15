import math
from typing import Any, Dict

import numpy as np
import pytest

from whylogs.core.dataset_profile import DatasetProfile
from whylogs.core.datatypes import DataType
from whylogs.core.metrics import Metric, MetricConfig
from whylogs.core.metrics.unicode_range import _STRING_LENGTH, UnicodeRangeMetric
from whylogs.core.preprocessing import PreprocessedColumn
from whylogs.core.resolvers import Resolver
from whylogs.core.schema import ColumnSchema, DatasetSchema


def test_unicode_range_metric() -> None:
    metric = UnicodeRangeMetric({"digits": (48, 57), "alpha": (97, 122)})
    strings = ["1", "12", "123", "1234a", "abc", "abc123"]
    col = PreprocessedColumn.apply(strings)
    digit_counts = [1, 2, 3, 4, 0, 3]
    alpha_counts = [0, 0, 0, 1, 3, 3]
    metric.columnar_update(col)

    assert metric.submetrics["digits"].mean.value == np.array(digit_counts).mean()
    assert metric.submetrics["alpha"].mean.value == np.array(alpha_counts).mean()
    assert metric.submetrics[_STRING_LENGTH].mean.value == np.array([len(s) for s in strings]).mean()


def test_unicode_range_metric_zero() -> None:
    metric = UnicodeRangeMetric.zero(MetricConfig())
    for range in MetricConfig().unicode_ranges.keys():
        assert range in metric.submetrics


@pytest.mark.parametrize(
    "bad_range",
    [
        {"bad": (-4, -1)},
        {"bad": (4, 1)},
        {"bad": (-1, 1)},
        {"bad": (1, 0x11FFFF)},
        {"very:bad": (1, 2)},
        {"also/bad": (1, 2)},
        {_STRING_LENGTH: (1, 2)},
    ],
)
def test_unicode_range_metric_invalid_initialization(bad_range):
    with pytest.raises(ValueError):
        UnicodeRangeMetric(bad_range)


def test_unicode_range_metric_serialization() -> None:
    metric = UnicodeRangeMetric({"digits": (48, 57), "alpha": (97, 122)})
    col = PreprocessedColumn.apply(["1", "12", "123", "1234a", "abc", "abc123"])
    metric.columnar_update(col)
    msg = metric.to_protobuf()
    deserialized = UnicodeRangeMetric.from_protobuf(msg)

    assert deserialized.submetrics["digits"].mean.value == metric.submetrics["digits"].mean.value
    assert deserialized.submetrics["alpha"].mean.value == metric.submetrics["alpha"].mean.value
    assert len(deserialized.submetrics) == len(metric.submetrics)


def test_unicode_range_metric_summary() -> None:
    metric = UnicodeRangeMetric({"digits": (48, 57), "alpha": (97, 122)})
    col = PreprocessedColumn.apply(["1", "12", "123", "1234a", "abc", "abc123"])
    metric.columnar_update(col)
    summary = metric.to_summary_dict(None)

    assert "unicode_range/digits/mean" in summary
    assert "unicode_range/alpha/mean" in summary
    assert f"unicode_range/{_STRING_LENGTH}/mean" in summary


def test_unicode_range_metric_merge() -> None:
    metric1 = UnicodeRangeMetric({"digits": (48, 57), "alpha": (97, 122)})
    metric2 = UnicodeRangeMetric({"digits": (48, 57), "alpha": (97, 122)})
    col1 = PreprocessedColumn.apply(["1", "12", "123"])
    col2 = PreprocessedColumn.apply(["1234a", "abc", "abc123"])
    metric1.columnar_update(col1)
    metric2.columnar_update(col2)
    merged = metric1 + metric2

    assert merged.submetrics["digits"].kll.value.get_n() == 6
    assert merged.submetrics["digits"].kll.value.get_min_value() == 0
    assert merged.submetrics["digits"].kll.value.get_max_value() == 4

    assert merged.submetrics["alpha"].kll.value.get_n() == 6
    assert merged.submetrics["alpha"].kll.value.get_min_value() == 0
    assert merged.submetrics["alpha"].kll.value.get_max_value() == 3


class UnicodeResolver(Resolver):
    def resolve(self, name: str, why_type: DataType, column_schema: ColumnSchema) -> Dict[str, Metric]:
        return {"unicode_range": UnicodeRangeMetric({"digits": (48, 57), "alpha": (97, 122)})}


class UnicodeSchema(DatasetSchema):
    types = {
        "col1": str,
    }
    resolvers = UnicodeResolver()


def _NaNfully_equal(left: Dict[Any, Any], right: Dict[Any, Any]) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    for key in left.keys():
        if (left[key] != right[key]) and not (math.isnan(left[key]) and math.isnan(right[key])):
            return False
    return True


def test_unicode_range_metric_in_profile() -> None:
    row = {"col1": "abc123"}
    schema = UnicodeSchema()
    prof = DatasetProfile(schema)
    prof.track(row=row)
    prof1_view = prof.view()
    prof1_view.write("/tmp/test_unicode_range_metric_in_profile")
    prof2_view = DatasetProfile.read("/tmp/test_unicode_range_metric_in_profile")
    prof1_cols = prof1_view.get_columns()
    prof2_cols = prof2_view.get_columns()

    assert prof1_cols.keys() == prof2_cols.keys()
    for col_name in prof1_cols.keys():
        col1_prof = prof1_cols[col_name]
        col2_prof = prof2_cols[col_name]
        assert (col1_prof is not None) == (col2_prof is not None)
        if col1_prof:
            assert col1_prof._metrics.keys() == col2_prof._metrics.keys()
            assert _NaNfully_equal(col1_prof.to_summary_dict(), col2_prof.to_summary_dict())
