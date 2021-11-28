from typing import Dict, Iterable, List, Tuple
import logging
import os

from overrides import overrides

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, ListField, TextField, SpanField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer

logger = logging.getLogger(__name__)

@DatasetReader.register('si-reader')
class SpanIdentificationReader(DatasetReader):
    def __init__(
            self,
            data_directory_path: str,
            tokenizer: Tokenizer = WhitespaceTokenizer(),
            token_indexers: Dict[str, TokenIndexer] = {"tokens": SingleIdTokenIndexer()},
            **kwargs
        ):
            super().__init__(**kwargs)
            self._data_directory_path = data_directory_path
            self.tokenizer = tokenizer
            self.token_indexers = token_indexers

    @overrides
    def text_to_instance(self, text: str, spans: List[Tuple[int, int]] = None) -> Instance:
        fields: Dict[str, Field] = {}

        # Get list of sequences
        sequences: List[TextField] = [TextField(self.tokenizer.tokenize(text[span_start, span_end]), self.token_indexers)
                                      for span_start, span_end in spans]

        # Add spans to instance
        list_span_fields: List[SpanField] = [SpanField(span[0], 
                                                       span[1], 
                                                       seq) 
                                                       for span, seq in zip(spans, sequences)]
        fields["spans"] = ListField(list_span_fields)

        # Add text to instance
        tokens = self.tokenizer.tokenize(text)
        fields["text"] = TextField(tokens, self.token_indexers)

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:

        logger.info(f"Searching for data in {self._data_directory_path}...")

        assert file_path.endswith(".labels"), ".labels file expected"

        with open(file_path, 'r') as lines:
            logger.info(f"Loading labels from {os.path.join(self._data_directory_path, file_path)}...")

            current_article_id: str = None
            spans: List[Tuple[int, int]] = []
            for line in lines:
                article_id, span_start, span_end = line.strip().split("\t")

                # Deal with first article
                if current_article_id == None:
                    current_article_id = article_id

                if article_id != current_article_id:
                    # New article, flush old article data
                    yield self._instanciate_article_data(current_article_id, spans)
                    spans.clear()
                    current_article_id = article_id

                # Remember spans for current article
                spans.append((span_start, span_end))

            # Deal with last article
            yield self._instanciate_article_data(current_article_id, spans)

    def _instanciate_article_data(self, article_id, spans):
        text = None
        with os.scandir(self._data_directory_path) as entries:
            for entry in entries:
                if article_id in entry.name and entry.name.endswith(".txt"):
                    with open(entry, 'r') as file:
                        text = file.read().replace('\n', ' ')
        assert text is not None, f"Article {article_id} not found"

        return self.text_to_instance(text, spans)