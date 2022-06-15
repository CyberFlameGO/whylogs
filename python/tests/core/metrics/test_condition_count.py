import re
from typing import Dict

import pandas as pd
import pytest

from whylogs.core.dataset_profile import DatasetProfile
from whylogs.core.datatypes import DataType
from whylogs.core.metrics import Metric
from whylogs.core.metrics.condition_count_metric import (
    ConditionCountConfig,
    ConditionCountMetric,
)
from whylogs.core.metrics.metric_components import SumIntegralComponent
from whylogs.core.preprocessing import PreprocessedColumn
from whylogs.core.resolvers import Resolver
from whylogs.core.schema import ColumnSchema, DatasetSchema


def test_condition_count_metric() -> None:
    conditions = {
        "alpha": re.compile("[a-zA-Z]+"),
        "digit": re.compile("[0-9]+"),
    }
    metric = ConditionCountMetric(conditions, SumIntegralComponent(0))
    strings = ["abc", "123", "kwatz", "314159", "abc123"]
    metric.columnar_update(PreprocessedColumn.apply(strings))
    summary = metric.to_summary_dict(None)

    assert set(summary.keys()) == {"total", "alpha", "digit"}
    assert summary["total"] == len(strings)
    assert summary["alpha"] == 3  # "abc123" matches since it's not fullmatch
    assert summary["digit"] == 2


def test_add_conditions_to_metric() -> None:
    conditions = {
        "alpha": re.compile("[a-zA-Z]+"),
    }
    metric = ConditionCountMetric(conditions, SumIntegralComponent(0))
    strings = ["abc", "123", "kwatz", "314159", "abc123"]
    metric.columnar_update(PreprocessedColumn.apply(strings))
    metric.add_conditions({"digit": re.compile("[0-9]+")})
    metric.columnar_update(PreprocessedColumn.apply(strings))
    summary = metric.to_summary_dict(None)

    assert set(summary.keys()) == {"total", "alpha", "digit"}
    assert summary["total"] == 2 * len(strings)
    assert summary["alpha"] == 2 * 3  # "abc123" matches since it's not fullmatch
    assert summary["digit"] == 2


def test_bad_condition_name() -> None:
    conditions = {
        "total": re.compile(""),
    }
    with pytest.raises(ValueError):
        ConditionCountMetric(conditions, SumIntegralComponent(0))

    metric = ConditionCountMetric({}, SumIntegralComponent(0))
    with pytest.raises(ValueError):
        metric.add_conditions({"total": re.compile("")})


def test_condition_count_in_profile() -> None:
    class TestResolver(Resolver):
        def resolve(self, name: str, why_type: DataType, column_schema: ColumnSchema) -> Dict[str, Metric]:
            return {"condition_count": ConditionCountMetric.zero(column_schema.cfg)}

    conditions = {
        "alpha": "[a-zA-Z]+",
        "digit": "[0-9]+",
    }
    config = ConditionCountConfig(conditions=conditions)
    resolver = TestResolver()
    schema = DatasetSchema(default_configs=config, resolvers=resolver)

    row = {"col1": ["abc", "123"]}
    prof = DatasetProfile(schema)
    prof.track(row=row)
    prof1_view = prof.view()
    prof1_view.write("/tmp/test_condition_count_metric_in_profile")
    prof2_view = DatasetProfile.read("/tmp/test_condition_count_metric_in_profile")
    prof1_cols = prof1_view.get_columns()
    prof2_cols = prof2_view.get_columns()

    assert prof1_cols.keys() == prof2_cols.keys()
    for col_name in prof1_cols.keys():
        col1_prof = prof1_cols[col_name]
        col2_prof = prof2_cols[col_name]
        assert (col1_prof is not None) == (col2_prof is not None)
        if col1_prof:
            assert col1_prof._metrics.keys() == col2_prof._metrics.keys()
            assert col1_prof.to_summary_dict() == col2_prof.to_summary_dict()
            assert {
                "condition_count/total",
                "condition_count/alpha",
                "condition_count/digit",
            } <= col1_prof.to_summary_dict().keys()


def test_condition_count_in_column_profile() -> None:
    conditions = {
        "alpha": "[a-zA-Z]+",
        "digit": "[0-9]+",
    }
    config = ConditionCountConfig(conditions=conditions)
    metric = ConditionCountMetric.zero(config)

    row = {"col1": ["abc", "123"]}
    frame = pd.DataFrame(data=row)
    prof = DatasetProfile()
    prof.track(pandas=frame)

    prof._columns["col1"].add_metric(metric)
    prof.track(pandas=frame)
    prof_view = prof.view()

    summary = prof_view.get_column("col1").to_summary_dict()
    assert summary["condition_count/total"] > 0
    assert summary["condition_count/alpha"] > 0
    assert summary["condition_count/digit"] > 0


def test_condition_count_in_dataset_profile() -> None:
    conditions = {
        "alpha": "[a-zA-Z]+",
        "digit": "[0-9]+",
    }
    config = ConditionCountConfig(conditions=conditions)
    metric = ConditionCountMetric.zero(config)

    row = {"col1": ["abc", "123"]}
    frame = pd.DataFrame(data=row)
    prof = DatasetProfile()
    prof.track(pandas=frame)

    prof.add_metric("col1", metric)
    prof.track(pandas=frame)
    prof_view = prof.view()

    summary = prof_view.get_column("col1").to_summary_dict()
    assert summary["condition_count/total"] > 0
    assert summary["condition_count/alpha"] > 0
    assert summary["condition_count/digit"] > 0
