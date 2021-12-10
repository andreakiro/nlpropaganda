import logging
from typing import Dict, Optional
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor
from allennlp.modules.span_extractors.self_attentive_span_extractor import SelfAttentiveSpanExtractor
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.training.metrics.fbeta_multi_label_measure import FBetaMultiLabelMeasure

import torch
import torch.nn.functional as F

from sklearn.metrics import f1_score

from allennlp.nn import util
from allennlp.data import TextFieldTensors
from allennlp.models.model import Model
from torch.overrides import get_overridable_functions

logger = logging.getLogger(__name__)

@Model.register("technique-classifier")
class TechniqueClassifier(Model):
    """
    This model implements the propaganda technique classification
    from a given set of text spans identified as propagandist.

    # Parameters

    input_size: `int` required.
        Input size to our classification model i.e. spans embedding size.
    output_size: `int` required.
        Output size to our classification i.e. number of propaganda techniques.
    classifier: `torch.nn` optional.
        PyTorch Neural Network model to perform classification task.
    """
    def __init__(
        self, 
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        context_layer: Seq2SeqEncoder,
        feature_size: int,
        max_span_width: int,
        output_size: int = 15,
        classifier: torch.nn = None,
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer
        self._metrics = FBetaMultiLabelMeasure()

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
        input_size = ese_output_dim + ase_output_dim

        if classifier is None:
            self._classifier = torch.nn.Linear(input_size, output_size)
        else:
            self._classifier = classifier

    def forward(
        self, # type: ignore
        content: TextFieldTensors,
        si_spans: torch.IntTensor,
        gold_labels: torch.IntTensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters
        si_spans: `torch.IntTensor` required.
            Batch of spans on which to perform training or prediction.
        gold_spans: `torch.IntTensor` optional.
            Batch of gold spans used to perform model training. 
        gold_labels: `torch.IntTensor` optional.
            Batch of gold_spans labels used to perform model training.

        # Returns 
        An output dictionary consisting of:
        technique: `torch.IntTensor` always
            The model prediction on the batch of given SpanFields.
        loss: `torch.FloatTensor` optional
            A scalar loss to be optimised when training.
        """
        # Shape: (batch_size, article_length, embedding_size)
        text_embeddings = self._text_field_embedder(content)

        # Shape: (batch_size, article_length)
        text_mask = util.get_text_field_mask(content)

        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(si_spans.float()).long()

        # Shape: (batch_size, article_length, encoding_dim)
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)
        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)
        # Shape: (batch_size, num_spans, embedding_size)
        attended_span_embeddings = self._attentive_span_extractor(text_embeddings, spans)

        # Shape: (batch_size, num_spans, embedding_size + 2 * encoding_dim + feature_size)
        span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)

        # Shape: (batch_size, num_spans, num_classes)
        logits = self._classifier(span_embeddings)
        technique_probs = F.softmax(logits, dim=2)

        # Shape: (batch_size, num_spans, 1)
        pred_labels = torch.argmax(technique_probs, dim=2)
        
        # technique_preds = torch.FloatTensor(technique_probs.shape).zero_()
        # technique_preds.scatter_(0, max_idx, 1)

        output_dict = {
            # "technique_preds": technique_preds,
            "technique_probs": technique_probs,
        }

        if gold_labels is not None:
            # self._metrics(pred_labels, gold_labels)
            # logger.info(f'SIZE OF LOGITS: {logits.shape}')
            # logger.info(f'SIZE OF GOLD LABELS: {gold_labels.shape}')
            # logger.info(f'SIZE OF LOGITS FLATTENED: {torch.flatten(logits, end_dim=1).shape}')
            # logger.info(f'SIZE OF GOLD LABELS FLATTENED: {torch.flatten(gold_labels).shape}')
            output_dict["loss"] = F.cross_entropy(torch.flatten(logits, end_dim=1), torch.flatten(gold_labels))        
        
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._metrics.get_metric(reset)