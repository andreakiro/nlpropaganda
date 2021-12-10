from typing import Optional

import logging
import torch
import itertools
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

    def __call__(self, prop_spans: torch.Tensor, gold_spans: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        for i, article_spans in enumerate(prop_spans):
            article_gold_spans = gold_spans[i]
            self._t_cardinality += article_gold_spans.size(dim=0)
            if article_spans.numel() == 0:
                continue
            merged_prop_spans = self._merge_intervals(article_spans)
            self._s_cardinality += merged_prop_spans.size(dim=0)
            for combination in itertools.product(merged_prop_spans, article_gold_spans):
                sspan = combination[0]
                tspan = combination[1]
                self._s_sum += self._c_function(sspan, tspan,
                                                sspan[1].item() - sspan[0].item() + 1)
                self._t_sum += self._c_function(sspan, tspan,
                                                tspan[1].item() - tspan[0].item() + 1)

    def get_metric(self, reset: bool = False):
        precision = 0
        recall = 0
        if self._s_cardinality != 0:
            precision = self._s_sum / self._s_cardinality
        if self._t_cardinality != 0:
            recall = self._t_sum / self._t_cardinality
        if reset:
            self.reset()
        return {"si-metric": (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0}

    def _c_function(self, s, t, h):
        """
        Compute C(s,t,h)
        :param s: predicted span 
        :param t: gold span
        :param h: normalizing factor
        :return: value of the function for the given parameters
        """
        intersection = self._intersect(s, t)
        return intersection / h if intersection > 0 else 0

    def _intersect(self, s, t):
        """
        Intersect two spans.
        :param s: first span
        :param t: second span
        :return: # of intersecting words between the spans, if < 0 represent the distance between spans, 
                 if = 0 the two words are neighbors
        """
        start = max(s[0].item(), t[0].item())
        end = min(s[1].item(), t[1].item())
        return end - start + 1

    def _merge_intervals(self, prop_spans):
        """
        Merge overlapping spans in the given span tensor.
        :param prop_spans: spans to be merged
        :return: tensor contained only non-overlapping spans
        """
        last = 0
        # Sort the predicted spans by start index
        prop_spans_sorted = torch.index_select(
            prop_spans, 0, torch.sort(prop_spans[:, 0])[1])
        # For each span in the sorted tensor, check for intersection with rightmost span analyzed
        to_delete = list()
        for index in range(1, prop_spans_sorted.size(dim=0)):
            if (self._intersect(prop_spans_sorted[index], prop_spans_sorted[last]) < 0):
                # add the span to the stack if it's empty or if the current span
                # does not intersect with the last span in the stack or has not distance 1 from it
                last += 1
            else:
                # current stack intersect with rightmost span analyzed
                # update last element right index with the max between the two intersecting
                prop_spans_sorted[last, 1] = max(
                    prop_spans_sorted[last, 1], prop_spans_sorted[index, 1])
                to_delete.append(index)

        mask = [x not in to_delete for x in torch.arange(
            prop_spans_sorted.size(dim=0))]
        return prop_spans_sorted[mask]
