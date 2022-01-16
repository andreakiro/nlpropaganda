from overrides import overrides

import numpy as np
import torch

from allennlp.data.fields.span_field import SpanField
from src.utils import int_to_label_alt
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import logging

@Predictor.register("tc-predictor-alt-demo")
class TechniqueClassificationPredictorAltDemo(Predictor):
    @overrides
    def predict_instance(
        self, 
        instance: Instance
    ) -> JsonDict:
        article_tokens = instance["content"]
        spans =  np.asarray(instance["spans"])
        output_dict = self._model.forward_on_instance(instance)

        with open('subs/output' + str("_tc_alt_demo_44040404040") + ".txt", 'a') as file:
            for span, technique_probs in zip(spans, output_dict["technique_probs"]):
                technique = int_to_label_alt(np.argmax(technique_probs))
                sp = span.human_readable_repr()
                toks = article_tokens[sp[0]:sp[1]+1]
                b = [t.text for t in toks]
                file.write(f"{' '.join(b)}\t{technique}\n")

        return None