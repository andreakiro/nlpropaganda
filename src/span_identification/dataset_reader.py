from typing import Dict, Iterable, List, Tuple
import logging
import os

from overrides import overrides

from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans
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
            max_span_width: int,
            tokenizer: Tokenizer = WhitespaceTokenizer(),
            token_indexers: Dict[str, TokenIndexer] = {"tokens": SingleIdTokenIndexer()},
            **kwargs
        ):
            super().__init__(**kwargs)
            self._data_directory_path = data_directory_path
            self._max_span_width = max_span_width
            self.tokenizer = tokenizer
            self.token_indexers = token_indexers

    @overrides
    def text_to_instance(self, text: str, gold_spans: List[Tuple[int, int]] = None) -> Instance:
        fields: Dict[str, Field] = {}

        # Create the TextField for the article
        tokens = self.tokenizer.tokenize(text)
        article_field = TextField(tokens, self.token_indexers)

        # Add gold spans to instance
        list_gold_span_fields: List[SpanField] = [SpanField(start, 
                                                            end - 1, # SpanField's bounds are inclusive, the dataset's end span is exclusive
                                                            article_field) 
                                                            for start, end in gold_spans]
        fields["gold_spans"] = ListField(list_gold_span_fields)

        # Extract article spans and add them to instance
        spans = enumerate_spans(tokens, max_span_width=self._max_span_width)
        list_span_fields: List[SpanField] = [SpanField(start, 
                                                       end, # enumerate_spans retruns spans with inclusive boundaries
                                                       article_field) 
                                                       for start, end in spans]
        #TODO: prune spans, by using heuristics and/or a model
        fields["spans"] = ListField(list_gold_span_fields)

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