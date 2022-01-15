from overrides import overrides

import numpy as np
import torch

from allennlp.data.fields.span_field import SpanField
from src.utils import int_to_label
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import logging
logger = logging.getLogger(__name__)

@Predictor.register("tc-predictor")
class TechniqueClassificationPredictor(Predictor):
    @overrides
    def predict_instance(
        self, 
        instance: Instance
    ) -> JsonDict:
        article_tokens = instance["content"]
        si_spans =  np.asarray(instance["si_spans"])
        output_dict = self._model.forward_on_instance(instance)
        article_id = output_dict["metadata"]["article_id"]

        with open('subs/output' + str("_tc_bert2005") + ".txt", 'a') as file:
            for span, technique_probs in zip(si_spans, output_dict["technique_probs"]):
                technique = int_to_label(np.argmax(technique_probs))
                sp = span.human_readable_repr()
                file.write(f"{article_id}\t{technique}\t{article_tokens[sp[0]].idx}\t{article_tokens[sp[1]].idx_end}\n")

        return None
