from overrides import overrides

import numpy as np
import torch

from allennlp.data.fields.span_field import SpanField
from src.utils import int_to_label_alt
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import logging
import re 



@Predictor.register("tc-predictor-alt")
class TechniqueClassificationPredictorAlt(Predictor):
    @overrides
    def predict_instance(
        self, 
        instance: Instance
    ) -> JsonDict:
        article_tokens = instance["content"]
        counter = 0
        spans =  np.asarray(instance["spans"])
        output_dict = self._model.forward_on_instance(instance)
        article_id = output_dict["metadata"]["article_id"]
        or_spans = output_dict["metadata"]["spans_or"]
        with open('submissions/output' + str("_tc_alt4444") + ".txt", 'a') as file:
            for span, technique_probs in zip(or_spans, output_dict["technique_probs"]):
                technique = int_to_label_alt(np.argmax(technique_probs))
                #sp = span.human_readable_repr()
                start = span[0]
                end = span[1]
                file.write(f"{article_id}\t{technique}\t{start}\t{end}\n")
        return None