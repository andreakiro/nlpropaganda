from overrides import overrides

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
        si_spans = output_dict["si_spans"]

        with open('submissions/output' + str(1) + ".txt", 'a') as file:
            for start, end in si_spans:
                file.write(f"{article_id}\t{article_tokens[start].idx}\t{article_tokens[end].idx_end}\n")

        return None
        # return super().predict_instance(instance)
