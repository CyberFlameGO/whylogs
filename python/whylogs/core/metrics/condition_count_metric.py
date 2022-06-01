import re
from dataclasses import dataclass
from typing import Any, Dict, List, Set

from whylogs.core.configs import SummaryConfig
from whylogs.core.metrics import Metric
from whylogs.core.metrics.metric_components import MetricComponent, SumIntegralComponent
from whylogs.core.metrics.metrics import MetricConfig, OperationResult
from whylogs.core.preprocessing import PreprocessedColumn
from whylogs.core.proto import MetricMessage


@dataclass(frozen=True)
class ConditionCountConfig(MetricConfig):
    conditions: Dict[str, str] = {}


@dataclass(frozen=True)
class ConditionCountMetric(Metric):
    conditions: Dict[str, Any]
    totals: Dict[str, SumIntegralComponent]
    matches: Dict[str, SumIntegralComponent]

    @property
    def namespace(self) -> str:
        return "condition_count"

    def get_component_paths(self) -> List[str]:
        paths: List[str] = []
        for cond_name in self.conditions.keys():
            paths += [f"{cond_name}/total", f"{cond_name}/matches"]

        return paths

    def columnar_update(self, data: PreprocessedColumn) -> OperationResult:
        if data.len <= 0:
            return OperationResult.ok(0)

        input = (
            data.pandas.strings.to_list() if data.pandas.strings is not None and not data.pandas.strings.empty else []
        )
        input = (input + data.list.strings) if data.list.strings else input
        for cond_name in self.conditions.keys():
            self.totals[cond_name].set(self.totals[cond_name].value + len(input))

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
            conditions={cond_name: re.compile(regex) for cond_name, regex in config.conditions.items()},
            totals={cond_name: SumIntegralComponent() for cond_name in config.conditions.keys()},
            matches={cond_name: SumIntegralComponent() for cond_name in config.conditions.keys()},
        )

    def to_protobuf(self) -> MetricMessage:
        msg = {}
        for cond_name in self.conditions.keys():
            msg[f"{cond_name}/total"] = self.totals[cond_name].to_protobuf()
            msg[f"{cond_name}/matches"] = self.matches[cond_name].to_protobuf()

        return MetricMessage(metric_components=msg)

    def to_summary_dict(self, cfg: SummaryConfig) -> Dict[str, Any]:
        summary = {}
        for cond_name in self.conditions.keys():
            summary[f"{cond_name}/total"] = (self.totals[cond_name].value,)
            summary[f"{cond_name}/matches"] = (self.matches[cond_name].value,)

        return summary

    @classmethod
    def from_protobuf(cls, msg: MetricMessage) -> "ConditionCountMetric":
        cond_names: Set[str] = set()
        for key, comp_msg in msg.metric_components.items():
            cond_name, comp_name = key.split("/")
            cond_names.add(cond_name)

        conditions = {cond_name: re.compile("") for cond_name in cond_names}
        totals = {cond_name: MetricComponent.from_protobuf(msg[f"{cond_name}/total"]) for cond_name in cond_names}
        matches = {cond_name: MetricComponent.from_protobuf(msg[f"{cond_name}/matches"]) for cond_name in cond_names}
        return ConditionCountMetric(
            conditions=conditions,
            totals=totals,
            matches=matches,
        )
