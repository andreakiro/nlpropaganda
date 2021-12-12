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
        output_dict = self._model.forward_on_instance(instance)
        article_id = output_dict["metadata"]["article_id"]
        si_tokens = output_dict["si_tokens"] # BUG

        logger.info(output_dict.keys())
        logger.info(output_dict["metadata"].keys())

        with open('submissions/output' + str(1) + ".txt", 'a') as file:
            for tokens in si_tokens:
                file.write(f"{article_id}\t{tokens[0].idx}\t{tokens[-1].idx_end}")

        return super().predict_instance(instance)
