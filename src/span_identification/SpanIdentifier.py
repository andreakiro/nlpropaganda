import logging
import math
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

from allennlp_models.coref.metrics.conll_coref_scores import ConllCorefScores
from allennlp_models.coref.metrics.mention_recall import MentionRecall

logger = logging.getLogger(__name__)

@Model.register("span-identifier")
class SpanIdentifier(Model):
    """
    TODO: UPDATE DESCRIPTION 
    This `Model` implements the coreference resolution model described in
    [Higher-order Coreference Resolution with Coarse-to-fine Inference](https://arxiv.org/pdf/1804.05392.pdf)
    by Lee et al., 2018.
    The basic outline of this model is to get an embedded representation of each span in the
    document. These span representations are scored and used to prune away spans that are unlikely
    to occur in a coreference cluster. For the remaining spans, the model decides which antecedent
    span (if any) they are coreferent with. The resulting coreference links, after applying
    transitivity, imply a clustering of the spans in the document.

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the `text` `TextField` we get as input to the model.
    context_layer : `Seq2SeqEncoder`
        This layer incorporates contextual information for each word in the document.
    classifier : `torch.nn`
        TODO Description of classifier.
    feature_size : `int`
        The embedding size for all the embedded features, such as distances or span widths.
    """
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        context_layer: Seq2SeqEncoder,
        classifier: torch.nn = None,
        feature_size: int,
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer

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

        ese_output_dim = _endpoint_span_extractor.get_output_dim()
        ase_output_dim = _attentive_span_extractor.get_output_dim()

        if classifier is None:
            self._classifier = torch.nn.Linear(ese_output_dim + ase_output_dim, 1)
        else self._classifier = classifier

    def forward(
        self,  # type: ignore
        text: TextFieldTensors,
        spans: torch.IntTensor,
        gold_spans: torch.IntTensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters TODO

        text : `TextFieldTensors`, required.
            The output of a `TextField` representing the text of
            the document.
        spans : `torch.IntTensor`, required.
            A tensor of shape (batch_size, num_spans, 2), representing the inclusive start and end
            indices of candidate spans for mentions. Comes from a `ListField[SpanField]` of
            indices into the text of the document.

        # Returns TODO

        An output dictionary consisting of:
        top_spans : `torch.IntTensor`
            A tensor of shape `(batch_size, num_spans_to_keep, 2)` representing
            the start and end word indices of the top spans that survived the pruning stage.
        antecedent_indices : `torch.IntTensor`
            A tensor of shape `(num_spans_to_keep, max_antecedents)` representing for each top span
            the index (with respect to top_spans) of the possible antecedents the model considered.
        predicted_antecedents : `torch.IntTensor`
            A tensor of shape `(batch_size, num_spans_to_keep)` representing, for each top span, the
            index (with respect to antecedent_indices) of the most likely antecedent. -1 means there
            was no predicted link.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised.
        """
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
            output_dict["loss"] = self._compute_loss(prop_spans, gold_spans)

        return output_dict

    # TODO
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        mention_recall = self._mention_recall.get_metric(reset)
        coref_precision, coref_recall, coref_f1 = self._conll_coref_scores.get_metric(reset)

        return {
            "coref_precision": coref_precision,
            "coref_recall": coref_recall,
            "coref_f1": coref_f1,
            "mention_recall": mention_recall,
        }

    def _compute_loss(
        self,
        prop_spans: torch.IntTensor,
        gold_spans: torch.IntTensor
    ) -> float:
    """
        Computes scores for every pair of spans. Additionally, a dummy label is included,
        representing the decision that the span is not coreferent with anything. For the dummy
        label, the score is always zero. For the true antecedent spans, the score consists of
        the pairwise antecedent score and the unary mention scores for the span and its
        antecedent. The factoring allows the model to blame many of the absent links on bad
        spans, enabling the pruning strategy used in the forward pass.
        # Parameters
        top_span_embeddings : `torch.FloatTensor`, required.
            Embedding representations of the kept spans. Has shape
            (batch_size, num_spans_to_keep, embedding_size)
        top_antecedent_embeddings: `torch.FloatTensor`, required.
            The embeddings of antecedents for each span candidate. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        top_partial_coreference_scores : `torch.FloatTensor`, required.
            Sum of span mention score and antecedent mention score. The coarse to fine settings
            has an additional term which is the coarse bilinear score.
            (batch_size, num_spans_to_keep, max_antecedents).
        top_antecedent_mask : `torch.BoolTensor`, required.
            The mask for valid antecedents.
            (batch_size, num_spans_to_keep, max_antecedents).
        top_antecedent_offsets : `torch.FloatTensor`, required.
            The distance between the span and each of its antecedents in terms of the number
            of considered spans (i.e not the word distance between the spans).
            (batch_size, num_spans_to_keep, max_antecedents).
        # Returns
        coreference_scores : `torch.FloatTensor`
            A tensor of shape (batch_size, num_spans_to_keep, max_antecedents + 1),
            representing the unormalised score for each (span, antecedent) pair
            we considered.
        """
        intersection = 0
        # for article_gs in gold_spans:
        #     for article_ps in prop_spans:
        #         for gs in article_gs:
        #             gs_start = gs[0].item()
        #             gs_end = gs[1].item()
        #             for ps in article_ps:
        #                 ps_start = ps[0].item()
        #                 ps_end = ps[1].item()
        #                 # case 1:
        #                 if gs_start < ps_start:
        #                     if gs_end < ps_start:
        #                         intersection = 0
        #                     else internsection = gs_end - ps_start + 1 # (ps_start - gs_end)
        #                 # case 2:
        #                 elif ps_start < gs_start
        return 0