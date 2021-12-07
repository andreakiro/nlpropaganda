from typing import Optional

import logging
import torch
import itertools
import numpy as np
from allennlp.training.metrics.metric import Metric

logger = logging.getLogger(__name__)

class SpanIdenficationMetric(Metric):
    
    def __init__(self) -> None:
        self.t_cardinality = 0
        self.s_cardinality = 0
        self.s_sum = 0
        self.t_sum = 0

    def reset(self) -> None:
        self.t_cardinality = 0
        self.s_cardinality = 0
        self.cs = 0
        self.ct = 0

    def __call__(self, prop_spans: torch.Tensor, gold_spans: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        for article in range(prop_spans.size(dim=0)):
            # merge overlapping predicted spans in the same article
            merged = self.merge_intervals(prop_spans[article])
            # update partial |T| value
            self.t_cardinality += merged.size(dim=0)
            # update partial |S| value
            self.s_cardinality += gold_spans[article, :, :].size(dim=1)
            # update partial C(s, t, h) values 
            for combination in itertools.product(merged[:, 1:3], gold_spans[article, :, 1:3]):
                tspan = combination[0]
                sspan = combination[1]
                # comput C(s,t,|s|) and add it to the partial sum
                self.s_sum += self.c_function(sspan, tspan, sspan[1].item()-sspan[0].item()+1)
                # comput C(s,t,|t|) and add it to the partial sum
                self.t_sum += self.c_function(sspan, tspan, tspan[1].item()-tspan[0].item()+1)
            break

    def get_metric(self, reset: bool = False):
        precision = 0
        recall = 0
        if self.s_cardinality != 0:
            precision = self.s_sum / self.s_cardinality
        if self.t_cardinality != 0:
            recall = self.t_sum / self.t_cardinality
        return { "CustomF1Score" : (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0 }

    def c_function(self, s, t, h):
        """
        Compute C(s,t,h)
        :param s: predicted span 
        :param t: gold span
        :param h: normalizing factor
        :return: value of the function for the given parameters
        """
        intersection = self.intersect(s, t)
        return intersection/h if intersection > 0 else 0    

    
    def intersect(self, s, t):
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

    def merge_intervals(self, prop_spans):
        """
        Merge overlapping spans in the given span tensor.
        :param prop_spans: spans to be merged
        :return: tensor contained only non-overlapping spans
        """
        last = 0
        # sort the predicted spans by start index
        prop_spans_sorted = prop_spans[prop_spans[:, 1].sort()[1]]
        # for each span in the sorted tensor, check for intersection with rightmost span analyzed
        to_delete = list()
        for index in range(1, prop_spans_sorted.size(dim=0)):
            if(self.intersect(prop_spans_sorted[index,1:3], prop_spans_sorted[last,1:3]) < 0):
                # add the span to the stack if it's empty or if the current span 
                # does not intersect with the last span in the stack or has not distance 1 from it
                last += 1
            else:
                # current stack intersect with rightmost span analyzed
                # update last element right index with the max between the two intersecting
                prop_spans_sorted[last,2] = max(prop_spans_sorted[last,2], prop_spans_sorted[index,2])
                to_delete.append(index)

        mask = [x not in to_delete for x in torch.arange(prop_spans_sorted.size(dim=0))]
        return prop_spans_sorted[mask]

# DEBUG
# x = [[[1203002, 12, 21], [1203002, 6, 15],  [1203002, 23, 45], [1203002, 80, 99]],
#      [[1223333, 12, 21], [1223333, 6, 15], [1223333, 23, 45], [1223333, 80, 99]]]
# x_tensor = torch.tensor(x)
# y = [[[1203002, 16, 21], [1203002, 44, 47], [1203002, 80, 99]], 
#      [[1223333, 34, 35], [1223333, 55, 59], [1223333, 80, 99]]]
# y_tensor = torch.tensor(y)

# lala=SpanIdenficationMetric()
# lala.__call__(x_tensor,y_tensor)
# print(lala.get_metric())