import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from whylogs.core.configs import SummaryConfig
from whylogs.core.metrics import Metric
from whylogs.core.metrics.metric_components import MetricComponent, SumIntegralComponent
from whylogs.core.metrics.metrics import MetricConfig, OperationResult
from whylogs.core.preprocessing import PreprocessedColumn
from whylogs.core.proto import MetricMessage


@dataclass(frozen=True)
class ConditionCountConfig(MetricConfig):
    conditions: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionCountMetric(Metric):
    conditions: Dict[str, Any]
    total: SumIntegralComponent = SumIntegralComponent(0)
    matches: Dict[str, SumIntegralComponent] = field(default_factory=dict)

    @property
    def namespace(self) -> str:
        return "condition_count"

    def __post_init__(self) -> None:
        super(type(self), self).__post_init__()
        if "total" in self.conditions.keys():
            raise ValueError("Condition cannot be named 'total'")

        for cond_name in self.conditions.keys():
            if cond_name not in self.matches:
                self.matches[cond_name] = SumIntegralComponent(0)

    def get_component_paths(self) -> List[str]:
        paths: List[str] = ["total", ] + list(self.conditions.keys())
        return paths

    def columnar_update(self, data: PreprocessedColumn) -> OperationResult:
        if data.len <= 0:
            return OperationResult.ok(0)

        input = (
            data.pandas.strings.to_list() if data.pandas.strings is not None and not data.pandas.strings.empty else []
        )
        input = (input + data.list.strings) if data.list.strings else input
        self.total.set(self.total.value + len(input))

        for str in input:
            for cond_name, condition in self.conditions.items():
                if condition.match(str):
                    self.matches[cond_name].set(self.matches[cond_name].value + 1)

        return OperationResult.ok(len(input))

    @classmethod
    def zero(cls, config: MetricConfig) -> "ConditionCountMetric":
        if config is None or not isinstance(config, ConditionCountConfig):
            raise ValueError("ConditionCountMetric.zero() requires ConditionCountConfig argument")

        return ConditionCountMetric(
            conditions={cond_name: re.compile(regex) for cond_name, regex in config.conditions.items()}
        )

    def to_protobuf(self) -> MetricMessage:
        msg = {"total": self.total.to_protobuf()}
        for cond_name in self.conditions.keys():
            msg[cond_name] = self.matches[cond_name].to_protobuf()

        return MetricMessage(metric_components=msg)

    def to_summary_dict(self, cfg: SummaryConfig) -> Dict[str, Any]:
        summary = {"total": self.total.value}
        for cond_name in self.conditions.keys():
            summary[cond_name] = self.matches[cond_name].value

        return summary

    @classmethod
    def from_protobuf(cls, msg: MetricMessage) -> "ConditionCountMetric":
        cond_names: Set[str] = set(msg.metric_components.keys())
        cond_names.remove("total")

        conditions = {cond_name: re.compile("") for cond_name in cond_names}
        total = MetricComponent.from_protobuf(msg.metric_components["total"])
        matches = {cond_name: MetricComponent.from_protobuf(msg.metric_components[cond_name]) for cond_name in cond_names}
        return ConditionCountMetric(
            conditions,
            total,
            matches,
        )
