from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.data.fields import ListFieldTensors
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.models import Model

@Model.register('span_identifier')
class SpanIdentifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: SpanExtractor,
                 encoder: Seq2SeqEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        self.classifier = torch.nn.Linear(embedder.get_output_dim(), 2)

    # Note that the signature of forward() needs to match that of field names
    def forward(
        self, spans: ListFieldTensors, gold_spans: ListFieldTensors = None
    ) -> Dict[str, torch.Tensor]:
        # Calculate loss
        loss = 0

        return {"loss": loss}