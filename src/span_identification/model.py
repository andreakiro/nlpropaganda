from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.data.fields import ListFieldTensors
from allennlp.modules.span_extractors import SpanExtractor, span_extractor
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.models import Model

@Model.register('span_identifier')
class SpanIdentifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 span_extractor: SpanExtractor,
                 encoder: Seq2SeqEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        self.span_extractor = span_extractor
        self.classifier = torch.nn.Linear(embedder.get_output_dim(), 2)

    # Note that the signature of forward() needs to match that of field names
    def forward(
        self, text: TextFieldTensors, spans: torch.Tensor, gold_spans: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Contextualise input sequence
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)

        # Embed the input
        token_embedder = Embedding(embedding_dim=embedding_dim, vocab=vocab)
        embedder = BasicTextFieldEmbedder({"tokens": token_embedder})
        embedded_tokens = embedder(tokens_tensor)
        print("shape of embedded_tokens", embedded_tokens.shape)
        print("shape of spans_tensor:", spans_tensor.shape)  # type: ignore
        
        span_extractor = EndpointSpanExtractor(input_dim=embedding_dim, combination="x,y")
        embedded_spans = span_extractor(embedded_tokens, spans_tensor)



        # Calculate loss
        loss = 0

        return {"loss": loss}