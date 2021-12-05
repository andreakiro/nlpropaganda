from typing import Optional
from overrides import overrides

import torch
from allennlp.training.metrics.metric import Metric

@Metric.register("tc-metric")
class TechniqueClassificationMetric(Metric):
    @overrides
    def __init__(self) -> None:
        super().__init__()
    
    @overrides
    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor]):
        return super().__call__(predictions, gold_labels, mask)

    @overrides
    def get_metric(self, reset: bool):
        return super().get_metric(reset)

    @overrides
    def reset(self) -> None:
        return super().reset()