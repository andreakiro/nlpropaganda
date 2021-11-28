from typing import Dict

import torch
import numpy
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.models import Model

@Model.register('span_identifier')
class SpanIdentifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)

    # Note that the signature of forward() needs to match that of field names
    def forward(
        self, tokens: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        print("tokens:", tokens)
        print("label:", label)

        return {}