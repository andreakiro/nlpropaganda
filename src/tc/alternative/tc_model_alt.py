import logging
from typing import Dict, List
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor
from allennlp.modules.span_extractors.self_attentive_span_extractor import SelfAttentiveSpanExtractor
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.training.metrics.fbeta_multi_label_measure import FBetaMeasure
from allennlp.common.checks import ConfigurationError

import torch
import torch.nn.functional as F
import numpy as np

from allennlp.nn import util
from allennlp.data import TextFieldTensors
from allennlp.models.model import Model

logger = logging.getLogger(__name__)


@Model.register("technique-classifier-alt")
class TechniqueClassifierAlt(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        context_layer: Seq2SeqEncoder,
        feature_size: int,
        max_span_width: int,
        output_size: int = 14,
        classifier: torch.nn = None,
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer
        self._metrics = FBetaMeasure(average='micro')

        self._endpoint_span_extractor = EndpointSpanExtractor(
            context_layer.get_output_dim(),
            combination="x,y",
            num_width_embeddings=max_span_width,
            span_width_embedding_dim=feature_size,
            bucket_widths=False,
        )

        self._attentive_span_extractor = SelfAttentiveSpanExtractor(
            input_dim=text_field_embedder.get_output_dim()
        )

        ese_output_dim = self._endpoint_span_extractor.get_output_dim()
        ase_output_dim = self._attentive_span_extractor.get_output_dim()
        input_size = ese_output_dim + ase_output_dim

        if classifier is None:
            self._classifier = torch.nn.Linear(input_size, output_size)
        else:
            self._classifier = classifier

    def forward(
        self,  #  type: ignore
        content: TextFieldTensors,
        spans: torch.IntTensor,
        gold_labels: torch.IntTensor = None,
        metadata: List[Dict[str, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        # logger.info(spans.size())
        # Shape: (batch_size, article_length, embedding_size)
        text_embeddings = self._text_field_embedder(content)

        logits = None

        # Shape: (batch_size, article_length)
        text_mask = util.get_text_field_mask(content)

        # Shape: (batch_size, article_length, encoding_dim)
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)

        try:
            # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
            endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)
            # Shape: (batch_size, num_spans, embedding_size)
            attended_span_embeddings = self._attentive_span_extractor(text_embeddings, spans)

            # Shape: (batch_size, num_spans, embedding_size + 2 * encoding_dim + feature_size)
            span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)

            # Shape: (batch_size, num_spans, num_classes)
            logits = self._classifier(span_embeddings)
        
        except ConfigurationError:
            logger.info("Hey buddy!")
            logits = torch.randn(spans.size()[0], spans.size()[1], 14, requires_grad=True)
            logger.info(logits)
            logger.info(gold_labels)
        
        technique_probs = F.softmax(logits, dim=2)

        output_dict = {
            "technique_probs": technique_probs,
            "metadata": metadata
        }

        if gold_labels is not None:
            self._metrics(logits, gold_labels)
            sum = 6128
            w = np.array([2123, 1058, 621, 493, 466, 294, 229, 209, 144, 129, 107, 107, 76, 72])/sum
            nw = [1 - (x / sum) for x in w]
            weights = torch.as_tensor(nw, dtype=float)
            output_dict["loss"] = F.cross_entropy(logits[0].float(), gold_labels[0].long(), weight=weights.float())

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._metrics.get_metric(reset)

#  try:
#             # Shape: (batch_size, article_length, embedding_size)
#             text_embeddings = self._text_field_embedder(content)

#             # Shape: (batch_size, article_length)
#             text_mask = util.get_text_field_mask(content)

#             # Shape: (batch_size, article_length, encoding_dim)
#             contextualized_embeddings = self._context_layer(text_embeddings, text_mask)
#             # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
#             endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)
#             # Shape: (batch_size, num_spans, embedding_size)
#             attended_span_embeddings = self._attentive_span_extractor(text_embeddings, spans)

#             # Shape: (batch_size, num_spans, embedding_size + 2 * encoding_dim + feature_size)
#             span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)

#             # Shape: (batch_size, num_spans, num_classes)
#             logits = self._classifier(span_embeddings)
#             technique_probs = F.softmax(logits, dim=2)

#             output_dict = {
#                 "technique_probs": technique_probs,
#                 "metadata": metadata
#             }

#             if gold_labels is not None:
#                 self._metrics(logits, gold_labels)
#                 sum = 6128
#                 w = np.array([2123, 1058, 621, 493, 466, 294, 229, 209, 144, 129, 107, 107, 76, 72])/sum
#                 nw = [1 - (x / sum) for x in w]
#                 weights = torch.as_tensor(nw, dtype=float)
#                 output_dict["loss"] = F.cross_entropy(logits[0].float(), gold_labels[0].long(), weight=weights.float())

#         except ConfigurationError: 
#             # Very ugly walk around (occurs for 10/370 samples)

#             rng = np.random.default_rng(12345)
#             lab = rng.integers(low=0, high=14)
#             probs = np.zeros(14)
#             probs[lab]= 1

#             output_dict = {
#                 "technique_probs": torch.as_tensor(probs),
#             }

#             if gold_labels is not None:
#                 inp = torch.randn(3, 5, requires_grad=True)
#                 tar = torch.randint(5, (3,), dtype=torch.int64)
#                 output_dict = {}
#                 output_dict["loss"] = F.cross_entropy(inp, tar.long())