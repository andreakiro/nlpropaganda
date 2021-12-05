import logging
from typing import Dict

import torch
import torch.nn.functional as F

from allennlp.nn import util
from allennlp.models.model import Model
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from src.si.si_metric import SpanIdenficationMetric

logger = logging.getLogger(__name__)

@Model.register("span-identifier")
class SpanIdentifier(Model):
    """
    This model implements the propaganda spans identification task
    from a given article and it's corresponding propaganda spans labels.

    # Parameters
    vocab: `Vocabulary` required.
        Model and architecture vocabulary.
    text_field_embedder : `TextFieldEmbedder` required.
        Used to embed the `text` `TextField` we get as input to the model.
    context_layer : `Seq2SeqEncoder` required.
        This layer incorporates contextual information for each word in the article.
    feature_size : `int` required.
        The embedding size for all the embedded features e-g. span widths.
    max_span_width : `int` required.
        Heuristic of the maximum width of candidate spans.
    classifier: `torch.nn` optional.
        PyTorch Neural Network model to perform span identification task.
    """
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
            combination = "x,y",
            num_width_embeddings = max_span_width,
            span_width_embedding_dim = feature_size,
            bucket_widths = False,
        )

        self._attentive_span_extractor = SelfAttentiveSpanExtractor(
            input_dim = text_field_embedder.get_output_dim()
        )

        ese_output_dim = self._endpoint_span_extractor.get_output_dim()
        ase_output_dim = self._attentive_span_extractor.get_output_dim()

        if classifier is None:
            self._classifier = torch.nn.Linear(ese_output_dim + ase_output_dim, 1)
        else:
            self._classifier = classifier

    def forward(
        self,  # type: ignore
        content: TextFieldTensors,
        all_spans: torch.IntTensor,
        gold_spans: torch.IntTensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters
        content: `TextFieldTensors` required.
            Batch of article's contents under textual format.
        all_spans: `torch.IntTensor` required.
            Batch of all spans retrieved by our DatasetReader
            on which we have to perform binary classification.
        gold_spans: `torch.IntTensor` required.
            Batch of gold spans i.e. the ground truth spans
            labelled as propaganda in the corresponding article.

        # Returns
        An output dictionary consisting of:
        si-spans: `torch.IntTensor` always.
            Batch of spans classified as propaganda by our model.
        probs-spans: `torch.IntTensor` always.
            Probability overview for every spans given as input.
        loss: `torch.FloatTensor` optional.
            A scalar loss to be optimised when training.
        """
        # Shape: (batch_size, article_length, embedding_size)
        text_embeddings = self._text_field_embedder(content)

        # Shape: (batch_size, article_length)
        text_mask = util.get_text_field_mask(content)

        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(all_spans.float()).long()

        # Shape: (batch_size, article_length, encoding_dim)
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)
        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)
        # Shape: (batch_size, num_spans, embedding_size)
        attended_span_embeddings = self._attentive_span_extractor(text_embeddings, spans)

        # Shape: (batch_size, num_spans, embedding_size + 2 * encoding_dim + feature_size)
        span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)

        # Shape: (batch_size, 1)
        logits = self._classifier(span_embeddings)
        probs = F.sigmoid(logits)

        # Shape: (batch_size, num_spans_propaganda, 2)
        si_spans = spans[probs >= 0.5]

        output_dict = {
            "si-spans": si_spans,
            "probs-spans": probs,
        }

        if gold_spans is not None:
            self._metrics(si_spans, gold_spans)
            output_dict["loss"] = self._metrics.get_metric()["si-loss"]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._metrics.get_metric(reset)