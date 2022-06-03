import re
from typing import Dict

from whylogs.core.dataset_profile import DatasetProfile
from whylogs.core.datatypes import DataType
from whylogs.core.metrics import Metric
from whylogs.core.metrics.condition_count_metric import ConditionCountConfig, ConditionCountMetric
from whylogs.core.preprocessing import PreprocessedColumn
from whylogs.core.resolvers import Resolver
from whylogs.core.schema import ColumnSchema, DatasetSchema


def test_condition_count_metric() -> None:
    conditions = {
        "alpha": re.compile("[a-zA-Z]+"),
        "digit": re.compile("[0-9]+"),
    }
    metric = ConditionCountMetric(conditions)
    strings = ["abc", "123", "kwatz", "314159", "abc123"]
    metric.columnar_update(PreprocessedColumn.apply(strings))
    summary = metric.to_summary_dict(None)

    assert set(summary.keys()) == {"total", "alpha", "digit"}
    assert summary["total"] == len(strings)
    assert summary["alpha"] == 3  # "abc123" matches since it's not fullmatch
    assert summary["digit"] == 2


def test_condition_count_in_profile() -> None:
    class TestResolver(Resolver):
        def resolve(self, name: str, why_type: DataType, column_schema: ColumnSchema) -> Dict[str, Metric]:
            return {"condition_count": ConditionCountMetric.zero(column_schema.cfg)}

    conditions = {
        "alpha": re.compile("[a-zA-Z]+"),
        "digit": re.compile("[0-9]+"),
    }
    config = ConditionCountConfig(conditions)
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
