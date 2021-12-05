from typing import Optional

import logging
import torch
from allennlp.training.metrics.metric import Metric

class SpanIdenficationMetric(Metric):

    def __init__(self, ) -> None:
        self.t_cardinality = 0
        self.s_cardinality = 0
        self.s_sum = 0
        self.t_sum = 0

    def reset(self) -> None:
        self.t = 0
        self.s = 0
        self.cs = 0
        self.ct = 0

    def __call__(self, prop_spans: torch.Tensor, gold_spans: torch.Tensor, mask: Optional[torch.BoolTensor]):
        self.t_cardinality += gold_spans.size(dim=1)
        self.s_cardinality += prop_spans.size(dim=1)
        for predicted_spans, true_spans in zip(prop_spans[article, :, :], gold_spans[article, :, :]):
            for combination in itertools.product(predicted_spans, true_spans):
                tspan = combination[0]
                sspan = combination[1]
                self.s_sum += c_function(sspan, tspan, sspan[1]-sspan[0])
                self.t_sum += c_function(sspan, tspan, tspan[1]-tspan[0])

    def get_metric(self, reset: bool = False):
        precision = 0
        recall = 0
        if s_cardinality != 0:
            precision = s_sum / s_cardinality
        if t_cardinality != 0:
            recall = t_sum / t_cardinality
        return (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    def c_function(s, t, h):
        start = max(s[0], t[0])
        end = min(s[1], t[1])
        return (end - start + 1)/h if end >= start else 0    

    def merge_intervals(prop_spans):
        stack = list()
        indices = prop_spans[:, :, 0].sort()[1] 
        a_sorted = a[torch.arange(a.size(0)).unsqueeze(1), indices] 
        logger.info(f"predictions: {token.text}")
        logger.info(f"sorted predictions: {token.text}")
        torch.stack(sorted(prop_spans, key=lambda a: a[0]))        

a = torch.randn(2, 5, 2)
indices = a[:, :, 0].sort()[1] 
a_sorted = a[torch.arange(a.size(0)).unsqueeze(1), indices] 
print(a, "\n\n") 
a = torch.stack(sorted(a, key=lambda a: a[:,:,0]))        
print("mio",a,"\n\n")
print(a_sorted)