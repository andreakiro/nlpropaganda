import logging
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

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
    This model implements the propaganda spans identification (SI) task
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
        logger.info("Initializing SpanIdentifier (SI) model...")

        # Define text embedder and encoder
        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer

        # Define first span extractor
        self._endpoint_span_extractor = EndpointSpanExtractor(
            context_layer.get_output_dim(),
            combination="x,y",
            num_width_embeddings=max_span_width,
            span_width_embedding_dim=feature_size,
            bucket_widths=False,
        )

        # Define second span extractor
        self._attentive_span_extractor = SelfAttentiveSpanExtractor(
            input_dim=text_field_embedder.get_output_dim()
        )

        # Define SI model metric
        self._metrics = SpanIdenficationMetric()

        # Define classifier architecture
        ese_output_dim = self._endpoint_span_extractor.get_output_dim()
        ase_output_dim = self._attentive_span_extractor.get_output_dim()
        input_dim = ese_output_dim + ase_output_dim
        output_dim = 1

        if classifier is None:
            self._classifier = torch.nn.Sequential(
                nn.Linear(input_dim, input_dim*2),
                nn.ReLU(),
                nn.Linear(input_dim*2, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, int(input_dim*0.6)),
                nn.ReLU(),
                nn.Linear(int(input_dim*0.6), int(input_dim*0.2)),
                nn.ReLU(),
                nn.Linear(int(input_dim*0.2), output_dim)
            )
        else:
            self._classifier = classifier

    def forward(
        self,  # type: ignore
        batch_content: TextFieldTensors,
        batch_all_spans: torch.IntTensor,
        batch_gold_spans: torch.IntTensor = None,
        metadata: List[Dict[str, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters
        batch_content: `TextFieldTensors` required.
            Batch of article's contents under textual format.
        batch_all_spans: `torch.IntTensor` required.
            Batch of all spans retrieved by our DatasetReader
            on which we have to perform binary classification.
        batch_gold_spans: `torch.IntTensor` required.
            Batch of gold spans i.e. the ground truth spans
            labelled as propaganda in the corresponding article.
        metadata: `List[Dict[str, int]]` optional.
            Metadata field containg information about weights
            to deal with important class inbalance of dataset.

        # Returns
        An output dictionary consisting of:
        all_spans: `torch.IntTensor` always.
            Batch of all spans extracted from our dataset.
        si_spans: `torch.IntTensor` always.
            Batch of spans classified as propaganda by our model.
        probs-spans: `torch.IntTensor` always.
            Probability overview for every spans given as input.
        loss: `torch.FloatTensor` optional.
            A scalar loss to be optimised when training.
        """
        # Shape: (batch_size, article_length, embedding_size)
        text_embeddings = self._text_field_embedder(batch_content)

        # Shape: (batch_size, article_length)
        text_mask = util.get_text_field_mask(batch_content)

        # Shape: (batch_size, num_spans, 2)
        batch_all_spans = F.relu(batch_all_spans.float()).long()

        # Shape: (batch_size, article_length, encoding_dim)
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)
        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, batch_all_spans)
        # Shape: (batch_size, num_spans, embedding_size)
        attended_span_embeddings = self._attentive_span_extractor(text_embeddings, batch_all_spans)

        # Shape: (batch_size, num_spans, embedding_size + 2 * encoding_dim + feature_size)
        span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)

        # Compute logits and probabilities
        # Shape: (batch_size, num_spans, 1)
        logits = self._classifier(span_embeddings)
        logits = torch.clamp(logits, min=-1e04, max=1e04)
        probs = torch.sigmoid(logits)

        # Create mask to filter spans
        mask = probs >= 0.5
        mask = torch.stack((mask, mask), dim=2)
        # Shape: (batch_size, num_spans_propaganda, 2)
        mask = mask.reshape(mask.shape[0], mask.shape[1], 2)

        # SI model prediction on propaganda spans
        batch_si_spans = torch.masked_select(batch_all_spans, mask)
        batch_si_spans = batch_si_spans.reshape(batch_all_spans.shape[0], -1, 2)

        output_dict = {
            "all_spans": batch_all_spans,
            "si_spans": batch_si_spans,
            "probs_spans": probs,
            "metadata": metadata,
        }

        if batch_gold_spans is not None:
            # During training mode
            weights = self._get_weights(metadata)
            target = self._get_target(probs, batch_all_spans, batch_gold_spans)

            # Compute SI metric and BCE loss
            self._metrics(batch_si_spans, batch_gold_spans)
            output_dict["loss"] = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weights)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._metrics.get_metric(reset)

    def _get_target(
        self,
        probs: torch.FloatTensor,
        batch_all_spans: torch.IntTensor,
        batch_gold_spans: torch.IntTensor,
    ) -> torch.FloatTensor:
        target = torch.zeros(probs.shape, dtype=torch.float).cuda()
        for i, spans in enumerate(batch_all_spans):
            if batch_gold_spans.numel() == 0:
                continue
            gold_spans = batch_gold_spans[i]
            for j, span in enumerate(spans):
                for gold_span in gold_spans:
                    if span[0] == gold_span[0] and span[1] == gold_span[1]:
                        target[i][j] = 1
                        break
        return target

    def _get_weights(
        self,
        metadata: List[Dict[str, int]]
    ) -> torch.FloatTensor:
        sum_spans = sum([data["num_all_spans"] for data in metadata])
        sum_gold_spans = sum([data["num_gold_spans"] for data in metadata])

        # Dirty fix
        if sum_gold_spans == 0:
            sum_gold_spans = 1

        return torch.tensor([np.sqrt(sum_spans / sum_gold_spans)]).cuda()
