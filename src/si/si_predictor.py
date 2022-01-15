from overrides import overrides

import numpy as np

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import logging
logger = logging.getLogger(__name__)

@Predictor.register("si-predictor")
class SpanIdentificationPredictor(Predictor):
    @overrides
    def predict_instance(
        self, 
        instance: Instance
    ) -> JsonDict:
        article_tokens = instance["batch_content"]
        output_dict = self._model.forward_on_instance(instance)
        article_id = output_dict["metadata"]["article_id"]
        si_spans = np.asarray(output_dict["all-spans"])
        probs = np.asarray(output_dict["probs"])

        mask = probs > 0.75
        mask = np.squeeze(mask, axis=1)

        si_spans = si_spans[mask, :]

        with open('subs/output_bert0' + str(123) + ".txt", 'a') as file:
            for start, end in si_spans:
                file.write(f"{article_id}\t{article_tokens[start].idx}\t{article_tokens[end].idx_end}\n")

        return None
        # return super().predict_instance(instance)
