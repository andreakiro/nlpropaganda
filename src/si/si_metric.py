from typing import Dict, List, Optional, Tuple

import logging
import torch
import itertools

import numpy as np

from allennlp.training.metrics.metric import Metric

logger = logging.getLogger(__name__)

class SpanIdenficationMetric(Metric):
    def __init__(self) -> None:
        self._s_cardinality = 0  #  S: model predicted spans
        self._t_cardinality = 0  # T: article gold spans
        self._s_sum = 0
        self._t_sum = 0

    def reset(self) -> None:
        self._s_cardinality = 0
        self._t_cardinality = 0
        self._s_sum = 0
        self._t_sum = 0

    def __call__(
        self,
        prop_spans: torch.Tensor, 
        gold_spans: torch.Tensor, 
        mask: Optional[torch.BoolTensor] = None
    ) -> None:
        for i in range(prop_spans.size(dim=0)):
            article_spans = prop_spans[i].numpy().tolist()
            article_gold_spans = gold_spans[i].numpy().tolist()

            self._t_cardinality += len(article_gold_spans)
            if len(article_spans) == 0:
                continue
            merged_prop_spans = self._merge_spans(article_spans)
            self._s_cardinality += len(merged_prop_spans)

            for combination in itertools.product(merged_prop_spans, article_gold_spans):
                sspan = combination[0]
                tspan = combination[1]
                self._s_sum += self._c_function(sspan, tspan, sspan[1] - sspan[0] + 1)
                self._t_sum += self._c_function(sspan, tspan, tspan[1] - tspan[0] + 1)

    def get_metric(
        self, 
        reset: bool = False
    ) -> Dict[str, int]:
        precision = 0
        recall = 0
        if self._s_cardinality != 0:
            precision = self._s_sum / self._s_cardinality
        if self._t_cardinality != 0:
            recall = self._t_sum / self._t_cardinality

        if reset:
            self.reset()
        return {"si-metric": (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0}

    def _c_function(
        self, 
        s: Tuple[int, int], 
        t: Tuple[int, int], 
        h: int
    ) -> int:
        """
        Compute C(s,t,h)
        :param s: predicted span 
        :param t: gold span
        :param h: normalizing factor
        :return: value of the function for the given parameters
        """
        intersection = self._intersect(s, t)
        return intersection / h if intersection > 0 else 0

    def _intersect(
        self, 
        s: Tuple[int, int], 
        t: Tuple[int, int]
    ) -> int:
        """
        Intersect two spans.
        :param s: first span
        :param t: second span
        :return: # of intersecting words between the spans, if < 0 represent the distance between spans, 
                 if = 0 the two words are neighbors
        """
        start = max(s[0], t[0])
        end = min(s[1], t[1])
        return end - start + 1

    def _merge_spans(
        self, 
        spans: List[List[int]]
    ) -> List[Tuple[int, int]]:
        """
        Merge overlapping spans in the given span tensor.
        :param prop_spans: spans to be merged
        :return: tensor contained only non-overlapping spans
        """
        sorted_spans = sorted(spans, key=lambda l: l[0])
        # For each span in the sorted list, check for intersection with rightmost span analyzed
        merged_spans = [sorted_spans[0]]
        for span in sorted_spans[1:]:
            # If the current interval does not overlap with the stack top, push it
            if span[0] > merged_spans[-1][1]:
                merged_spans.append((span[0], span[1]))
            # If the current interval overlaps with stack top and ending time of current interval is more than that of stack top, 
            # update stack top with the ending time of current interval
            elif span[1] >  merged_spans[-1][1]:
                merged_spans[-1] = (merged_spans[-1][0], span[1])
        return merged_spans
