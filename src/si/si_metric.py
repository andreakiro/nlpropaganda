from typing import Optional
from overrides import overrides

import torch
from allennlp.training.metrics.metric import Metric

@Metric.register("si-metric")
class SpanIdenficationMetric(Metric):
    @overrides
    def __init__(self, ) -> None:
        super().__init__()
    
    @overrides
    def __call__(self, prop_spans: torch.Tensor, gold_spans: torch.Tensor, mask: Optional[torch.BoolTensor]):
        return super().__call__(prop_spans, gold_spans, mask)

    @overrides
    def get_metric(self, reset: bool):
        metrics = {
            #TODO: compute loss!
            "si-loss": 0, 
        }

        return super().get_metric(reset)

    @overrides
    def reset(self) -> None:
        return super().reset()