import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward, GatedSum
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from allennlp.nn import util, InitializerApplicator
from src.si.si_metric import SpanIdenficationMetric

logger = logging.getLogger(__name__)

@Model.register("span-identifier")
class SpanIdentifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        context_layer: Seq2SeqEncoder,
        feature_size: int,
        max_span_width: int,
        classifier: torch.nn = None,
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer
        self._metrics = SpanIdenficationMetric()

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

        if classifier is None:
            self._classifier = torch.nn.Linear(ese_output_dim + ase_output_dim, 1)
        else:
            self._classifier = classifier

    def forward(
        self,  # type: ignore
        text: TextFieldTensors,
        spans: torch.IntTensor,
        gold_spans: torch.IntTensor = None,
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, article_length, embedding_size)
        text_embeddings = self._text_field_embedder(text)

        batch_size = spans.size(0)
        article_length = text_embeddings.size(1)
        num_spans = spans.size(1)

        # Shape: (batch_size, article_length)
        text_mask = util.get_text_field_mask(text)

        # SpanFields return -1 when they are used as padding. As we do
        # some comparisons based on span widths when we attend over the
        # span representations that we generate from these indices, we
        # need them to be <= 0. This is only relevant in edge cases where
        # the number of spans we consider after the pruning stage is >= the
        # total number of spans, because in this case, it is possible we might
        # consider a masked span.

        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()

        # Shape: (batch_size, article_length, encoding_dim)
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)
        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)
        # Shape: (batch_size, num_spans, embedding_size)
        attended_span_embeddings = self._attentive_span_extractor(text_embeddings, spans)

        # Shape: (batch_size, num_spans, embedding_size + 2 * encoding_dim + feature_size)
        span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)

        # TODO: Pruning !

        # Shape: (batch_size, 1)
        logits = self._classifier(span_embeddings)

        # Shape: (batch_size, 1)
        probs = F.sigmoid(logits)

        # Shape: (batch_size, num_spans_propaganda, 2)
        prop_spans = spans[probs >= 0.5]

        output_dict = {
            "propaganda_spans": prop_spans,
        }

        if gold_spans is not None:
            self._metrics(prop_spans, gold_spans)
            output_dict["loss"] = self._metrics.get_metric()["si-loss"]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._metrics.get_metric()