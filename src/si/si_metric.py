from typing import Optional

import logging
import torch
from allennlp.training.metrics.metric import Metric

class SpanIdenficationMetric(Metric):
    
    def __init__(self) -> None:
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
        for article in range(prop_spans.size(dim=0)):
            # merge overlapping predicted spans in the same article
            prop_spans[article] = merge_intervals(prop_spans[article])
            for combination in itertools.product(prop_spans[article, :, :], gold_spans[article, :, :]):
                tspan = combination[0]
                sspan = combination[1]
                # comput C(s,t,|s|) and add it to the partial sum
                self.s_sum += c_function(sspan, tspan, sspan[1]-sspan[0])
                # comput C(s,t,|t|) and add it to the partial sum
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
        """
        Compute C(s,t,h)
        :param s: predicted span 
        :param t: gold span
        :param h: normalizing factor
        :return: value of the function for the given parameters
        """
        intersection = intersect(s, t)
        return intersection/h if intersection > 0 else 0    

    
    def intersect(s, t):
        """
        Intersect two spans.
        :param s: first span
        :param t: second span
        :return: # of intersecting words between the spans, if < 0 represent the distance between spans
        """
        start = max(s[0], t[0])
        end = min(s[1], t[1])
        return end - start + 1

    def merge_intervals(prop_spans):
        """
        Merge overlapping spans in the given span tensor.
        :param prop_spans: spans to be merged
        :return: tensor contained only non-overlapping spans
        """
        last = 0
        # sort the predicted spans by start index
        prop_spans[prop_spans[:, 0].sort()[1]]
        # debug
        logger.info(f"please check me!")
        logger.info(f"predictions: {prop_spans}")
        logger.info(f"sorted predictions: {prop_spans_sorted}")
        # for each span in the sorted tensor, check for intersection with rightmost span analyzed
        for index in range(prop_spans_sorted.size(dim=0)):
            if((not last) or (intersect(prop_spans_sorted[index,:], prop_spans_sorted[last,:]) < -1)):
                # add the span to the stack if it's empty or if the current span 
                # does not intersect with the last span in the stack or has not distance 1 from it
                last += 1
            else:
                # current stack intersect with rightmost span analyzed
                # update last element right index with the max between the two intersecting
                prop_spans_sorted[last,1] = max(prop_spans_sorted[last,1], prop_spans_sorted[index,1])
                # remove the other from the tensor
                logger.info(f"please check me!")
                logger.info(f"size before deletion: {prop_spans_sorted.size(dim=0)}")
                prop_spans_sorted = prop_spans_sorted[torch.arange(prop_spans_sorted.size(dim=0))!=index]
                logger.info(f"size after deletion: {prop_spans_sorted.size(dim=0)}")

        return prop_spans_sorted