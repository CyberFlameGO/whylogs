import re
from typing import Dict

from whylogs.core.dataset_profile import DatasetProfile
from whylogs.core.datatypes import DataType
from whylogs.core.metrics import Metric
from whylogs.core.metrics.merics import ConditionCountConfig, ConditionCountMetric
from whylogs.core.resolvers import Resolver
from whylogs.core.schema import ColumnSchema, DatasetSchema


def test_condition_count_in_profile() -> None:
    class TestResolver(Resolver):
        def resolve(self, name: str, why_type: DataType, column_schema: ColumnSchema) -> Dict[str, Metric]:
            return {"condition_count": ConditionCountMetric.zero(column_schema.cfg)}

    conditions = {
        "alpha": re.compile("[a-zA-Z]+"),
        "digit": re.compile("[0-9]+"),
    }
    config = ConditionCountConfig(conditions)
    resolver = TestResolver(ColumnSchema(cfg=config))
    schema = DatasetSchema(resolvers=resolver, default_configs=config)

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
